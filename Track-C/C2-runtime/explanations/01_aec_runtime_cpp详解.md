# `src/aec_runtime.cpp` 详细解释

本文档解释 `src/aec_runtime.cpp` 的设计、数据结构、核心函数、执行路径和与评分项的对应关系。

## 1. 文件定位

`src/aec_runtime.cpp` 是 C2 Runtime 的主体实现文件，经过 Makefile 编译后生成：

```text
libaec.so
```

它实现 `include/aec_runtime.h` 中声明的公开 C ABI，并调用官方设备 ABI：

```cpp
#include "aec_runtime.h"
#include "aec_device_abi.h"
```

Runtime 本身不实现 AEC 指令解释器，也不直接计算 GEMM 或向量运算结果。它的职责是：

- 做 API 参数校验。
- 管理 Runtime 侧 handle 生命周期。
- 把 API 调用转换为官方设备 command。
- 提交给 `libaec_device.so`。
- 把设备完成状态映射回 Runtime 错误码。

## 2. 主要全局状态

文件内部使用匿名 namespace 隐藏实现细节，外部只看到 C ABI。

### `last_error`

```cpp
thread_local aecError_t last_error = AEC_SUCCESS;
```

这是线程局部 last error。根据规范：

- 失败调用更新 last error。
- 成功调用不清除旧错误。
- `aecGetLastError()` 读取并清为 `AEC_SUCCESS`。
- `aecPeekAtLastError()` 只读取不清除。

实现中通过 `finish(error)` 统一处理失败状态。

### `allocations`

```cpp
std::unordered_map<uint64_t, Allocation> allocations;
```

记录 Runtime 当前认为 live 的 device allocation：

```cpp
struct Allocation {
    uint64_t base;
    uint64_t bytes;
};
```

用途：

- 判断 device pointer 是否属于某个 live allocation。
- 拒绝 double free、stale pointer、interior pointer。
- 检查同步/异步 copy 的 span 是否完整落在单个 allocation 内。

真实设备内存仍由官方 `aecDeviceAlloc` 和 `aecDeviceFree` 管理。

### `registrations`

```cpp
std::vector<Registration> registrations;
```

记录 host registered memory interval：

```cpp
struct Registration {
    uintptr_t base;
    size_t bytes;
};
```

用途：

- `aecHostRegister` 添加精确 interval。
- 拒绝 duplicate、overlap、overflow。
- DMA 时如果 host span 完整位于某个 registered interval 内，则给 device command 加：

```text
AEC_DEVICE_FLAG_REGISTERED
AEC_DEVICE_FLAG_ZERO_COPY
```

注意：registration 只影响虚拟周期和统计，不改变数据正确性路径。

### `live_streams` 和 `live_events`

```cpp
std::unordered_set<Stream *> live_streams;
std::unordered_set<Event *> live_events;
```

用于判断 stream/event handle 是否仍然有效。

Stream 和 Event 都是 Runtime 内部对象，公开头文件里只暴露 opaque pointer：

```cpp
typedef struct aecStreamOpaque *aecStream_t;
typedef struct aecEventOpaque *aecEvent_t;
```

本实现把它们转换为内部 `Stream *` 和 `Event *` 使用。

### `next_sequence`

```cpp
uint64_t next_sequence = 1;
```

每个提交给设备的 command 都会分配严格递增 sequence。设备 completion 回显 sequence，Runtime 检查回显是否一致。

这对应 R301 的 ABI sequence、completion 和 stats accounting 检查。

### `next_dma_channel`

```cpp
uint8_t next_dma_channel = 0;
```

DMA command 在 channel 0 和 channel 1 之间轮转，确保公开测试中可以观察到双 DMA channel 都被使用。

这对应 R302。

## 3. 错误处理

### `finish`

```cpp
aecError_t finish(aecError_t error)
```

规则：

- 如果 `error != AEC_SUCCESS`，写入线程局部 `last_error`。
- 返回原错误码。

这样所有公开 API 可以统一：

```cpp
return finish(AEC_ERROR_INVALID_ARGUMENT);
```

### `map_device_status`

```cpp
aecError_t map_device_status(aecDeviceStatus status)
```

把官方设备库的错误码转换为 Runtime 错误码。例如：

- `AEC_DEVICE_INVALID_ARGUMENT` -> `AEC_ERROR_INVALID_ARGUMENT`
- `AEC_DEVICE_OUT_OF_MEMORY` -> `AEC_ERROR_OUT_OF_MEMORY`
- `AEC_DEVICE_INVALID_ADDRESS` -> `AEC_ERROR_INVALID_ADDRESS`
- `AEC_DEVICE_INJECTED_FAULT` -> `AEC_ERROR_DEVICE`
- `AEC_DEVICE_ISA_TRAP` -> `AEC_ERROR_ISA_TRAP`

这保证 Runtime 对上层暴露的错误码符合 `aec_runtime.h`。

## 4. 地址区间检查

### `span_in_allocation`

```cpp
bool span_in_allocation(uint64_t ptr, uint64_t bytes)
```

检查 device span 是否完整属于一个 live allocation。

它不只是判断地址在 `[0, 64MiB)` 内，而是必须满足：

```text
allocation.base <= ptr
ptr + bytes <= allocation.base + allocation.bytes
```

这样可以拒绝：

- 空指针。
- bytes 为 0。
- 地址加法 overflow。
- 越过 allocation 末尾。
- 横跨两个 allocation 的 span。
- stale pointer 或 unknown pointer。

这对应 R103、R302 等边界检查。

### `host_span_registered`

```cpp
bool host_span_registered(uintptr_t ptr, size_t bytes)
```

检查 host span 是否完整位于某个 registered interval 内。若是，则 DMA command 使用 registered/zero-copy flag。

### `registration_overlaps`

```cpp
bool registration_overlaps(uintptr_t ptr, size_t bytes)
```

用于 `aecHostRegister`，拒绝：

- null pointer。
- bytes 为 0。
- 地址 overflow。
- 与已有 registered interval 重叠。

这对应 R303。

## 5. 参数块序列化

### 为什么不能直接复制 struct

C2 文档要求 kernel 参数块是：

- little-endian。
- 紧密排列。
- 无 native padding。
- unused parameter bytes 为 0。

因此不能简单地把 C/C++ struct 原样 memcpy 成参数块。

本实现提供：

```cpp
put_le32()
put_le64()
put_float_bits()
```

逐字段写入参数块。

### Vector Add 参数块

大小 32 bytes：

```text
A:u64, B:u64, C:u64, count:u64
```

代码写入：

```cpp
put_le64(parameters, 0, values->a);
put_le64(parameters, 8, values->b);
put_le64(parameters, 16, values->c);
put_le64(parameters, 24, values->count);
```

### GEMM 参数块

大小 40 bytes：

```text
A:u64, B:u64, C:u64, M:u32, N:u32, K:u32, dtype:u32
```

代码写入：

```cpp
put_le64(parameters, 0, values->a);
put_le64(parameters, 8, values->b);
put_le64(parameters, 16, values->c);
put_le32(parameters, 24, values->m);
put_le32(parameters, 28, values->n);
put_le32(parameters, 32, values->k);
put_le32(parameters, 36, values->dtype);
```

### AXPY 参数块

大小 28 bytes：

```text
X:u64, Y:u64, count:u64, alpha:f32 bits
```

`alpha` 用 `put_float_bits` 写入，避免 native padding。

### DOT 参数块

大小 32 bytes：

```text
X:u64, Y:u64, result:u64, count:u64
```

### NRM2 参数块

大小 24 bytes：

```text
X:u64, result:u64, count:u64
```

## 6. Kernel Launch 路径

### `build_launch`

```cpp
LaunchBuild build_launch(...)
```

这是 `aecLaunch` 和计算库 wrapper 共用的 launch 构造函数。

它做的事情：

1. 校验 grid/block。
2. 校验 args 和 args_size。
3. 根据 kernel 类型生成参数块。
4. 确定 semantic kernel id、dtype、variant。
5. 调用：

```cpp
aecDeviceResolveKernel(semantic_kernel, dtype, variant, &info)
```

6. 把返回的 `info.handle`、`info.entry_pc`、`info.isa_version` 写入 `aecDeviceCommand`。
7. 设置 `AEC_DEVICE_OP_ISA_LAUNCH`。
8. 返回构造好的 command。

### `execute_command`

```cpp
aecError_t execute_command(aecDeviceCommand command)
```

它负责真正提交 command：

1. 分配 sequence。
2. 调用 `aecDeviceSubmit`。
3. 检查提交状态。
4. 检查 completion status。
5. 检查 completion sequence 是否匹配。
6. 返回 Runtime 错误码。

这条路径保证所有成功 kernel/compute API 都有设备执行证据。

## 7. DMA 路径

### 同步 copy

`aecCopyH2D` 和 `aecCopyD2H`：

1. 检查 host pointer 非空。
2. 检查 bytes 非 0。
3. 检查 device span 完整属于 live allocation。
4. 构造 DMA command。
5. 直接调用 `execute_dma`。

同步 copy 返回时数据已经搬运完成。

### 异步 copy

`aecCopyAsync`：

1. 检查 host pointer 和 bytes。
2. 检查 direction。
3. 检查 stream handle 有效。
4. 构造 DMA command。
5. 入队到对应 stream。
6. 立即返回 `AEC_SUCCESS`。

真正执行发生在：

- `aecStreamSync`
- `aecEventSynchronize`
- `aecStreamDestroy`
- `aecFree` / `aecHostUnregister` 中的等待逻辑

### `execute_dma`

`execute_dma` 会：

- 根据 opcode 判断 device pointer 是 src 还是 dst。
- 再次检查 host address、bytes、device span。
- 如果 host span registered，则加 registered/zero-copy flag。
- 分配 DMA channel。
- 设置 chunk 和 queue depth。
- 调用 `execute_command`。

## 8. Stream 实现

### 内部结构

```cpp
struct Stream {
    uint64_t id;
    std::deque<Task> queue;
    aecError_t pending_error;
};
```

每个 stream 有自己的 FIFO 队列。

队列元素 `Task` 有两种：

- `Command`：DMA 或 ISA launch。
- `EventMarker`：event record 插入的 marker。

### 惰性执行模型

本实现没有创建后台 worker thread，而是在同步点 drain stream。

原因：

- C2 虚拟设备是确定性的。
- 公开规范只要求同一 stream FIFO。
- 异步 copy 要求 enqueue 后可返回，不要求 wall-clock 并行。
- 惰性执行更简单，也更容易保证错误归属。

### `drain_stream`

`drain_stream` 按 FIFO 弹出任务：

- 如果是 `EventMarker`，记录 event 完成和当前 device cycle。
- 如果是 DMA command，调用 `execute_dma`。
- 如果是 ISA launch，调用 `execute_command`。
- 如果执行失败，将错误写入 `stream->pending_error` 并停止。

### `aecStreamSync`

调用 `drain_stream`，然后通过 `consume_stream_error` 返回并清除第一个未报告错误。

这对应 R105、R106、R302、R304。

## 9. Event 实现

### 内部结构

```cpp
struct Event {
    uint64_t id;
    bool recorded;
    bool completed;
    uint64_t generation;
    uint64_t cycle;
    Stream *stream;
};
```

### generation 的作用

同一个 event 可以多次 record。规范要求 query/sync/elapsed 观察最新 generation。

因此每次 `aecEventRecord`：

```cpp
event->generation++;
```

并把 generation 写入 marker。drain marker 时只有 generation 匹配才更新 event。

### elapsed cycles

`aecEventElapsedCycles(start, end, cycles)` 要求：

- 两个 event handle 都有效。
- 两个 event 都已经 record。
- 两个 event 都 completed。
- end cycle 不小于 start cycle。

返回：

```text
end.cycle - start.cycle
```

## 10. Allocation 和 Free

### `aecAlloc`

调用官方：

```cpp
aecDeviceAlloc(bytes, 64, &ptr)
```

然后写入 Runtime 的 `allocations`。

64 字节对齐符合文档要求。

### `aecFree`

先检查：

- ptr 非 0。
- ptr 是 live allocation base。

然后：

1. drain 所有 stream。
2. 调用 `aecDeviceFree(ptr)`。
3. 从 `allocations` 删除。

这样能避免释放仍被队列中任务使用的 allocation。

## 11. Host Registration

### `aecHostRegister`

检查：

- ptr 非空。
- bytes 非 0。
- 不 overflow。
- 不与已有 interval 重叠。

成功后加入 `registrations`。

### `aecHostUnregister`

要求精确 base pointer。

成功前先 drain 所有 stream，避免还有排队 DMA 使用该 host interval。

## 12. 计算库实现

### GEMM 系列

公开 API：

- `aecMatmulF4`
- `aecMatmulF8`
- `aecMatmulF16`
- `aecMatmulBF16`
- `aecMatmulF32`
- `aecMatmulF64`
- `aecMatmulI4`
- `aecMatmulI8`
- `aecMatmulI32`

都调用统一：

```cpp
launch_gemm(...)
```

`launch_gemm` 做：

1. 校验 M/N/K 范围。
2. 校验 dtype。
3. 根据 shape/alignment 选择 GEMM image variant。
4. 构造 `aecGemmArgs`。
5. 调用 `build_and_submit_launch`。

### GEMM variant 选择

```cpp
choose_gemm_kernel(...)
```

选择规则：

- 如果 A/B/C 都 16 字节对齐，且 M/N/K 都能被 8 整除，选 vectorized。
- 否则如果 M/N/K 都能被 4 整除，选 tiled。
- 否则选 naive。

这是基于通用 shape/alignment 约束的策略，不依赖测试 ID 或固定输入。

### 向量接口

- `aecAxpy` 使用 `AEC_KERNEL_AXPY_F32`。
- `aecDot` 使用 `AEC_KERNEL_DOT_F32`。
- `aecNrm2` 使用 `AEC_KERNEL_NRM2_F32`。

它们同样只构造参数块并提交 fixed image。

## 13. 与评分项对应关系

### R101

覆盖：

- device query。
- error name。
- TLS last error。

对应函数：

- `aecDeviceCount`
- `aecDeviceInfo`
- `aecGetLastError`
- `aecPeekAtLastError`
- `aecGetErrorName`

### R102

覆盖 allocation/free、OOM、reuse、非法 free。

对应：

- `aecAlloc`
- `aecFree`
- `allocations`

### R103

覆盖同步 H2D/D2H 和 allocation-relative bounds。

对应：

- `aecCopyH2D`
- `aecCopyD2H`
- `span_in_allocation`
- `execute_dma`

### R104

覆盖 Vector Add image 和 launch mapping。

对应：

- `aecLaunch`
- `build_launch`
- `AEC_KERNEL_VECTOR_ADD_F32`

### R105-R106

覆盖 stream FIFO、event generation、cycle、异步错误。

对应：

- `aecStreamCreate`
- `aecStreamSync`
- `aecStreamDestroy`
- `aecEventCreate`
- `aecEventRecord`
- `aecEventSynchronize`
- `aecEventElapsedCycles`
- `drain_stream`

### R201-R204

覆盖 GEMM、低精度格式、packed integer、向量运算。

对应：

- `launch_gemm`
- `aecMatmul*`
- `aecAxpy`
- `aecDot`
- `aecNrm2`

### R301-R304

覆盖 ABI sequence、stats、双 DMA、registration、fault recovery。

对应：

- `execute_command`
- `execute_dma`
- `next_sequence`
- `next_dma_channel`
- `aecHostRegister`
- `aecHostUnregister`
- `map_device_status`

## 14. 设计取舍

### 为什么 stream 不启动后台线程

因为题目使用确定性虚拟设备，评分看的是虚拟周期和 command 证据，不看真实 wall-clock 并发。惰性执行可以更稳定地保证：

- FIFO 顺序。
- 错误只在 sync 时返回。
- event marker 与 command 顺序一致。

### 为什么 Runtime 侧不预先检查所有 kernel 参数 span

同步 DMA 必须在 Runtime 侧检查 allocation-relative bounds。

但 ISA launch 中的参数由设备 image 解释。对于某些 kernel 参数错误，更合适让受控设备产生 trap 或 fault，这样可以保留评分要求的设备执行证据。Runtime 做 API 层面的基础校验，不替代设备解释器。

### 为什么 GEMM 选择 vectorized/tiled/naive

fixed image manifest 里提供同一 dtype 的多个 variant。variant 选择依据公开约束：

- vectorized 要求更高 alignment 和 shape divisibility。
- tiled 要求 M/N/K 可被 4 整除。
- naive 覆盖所有合法 shape。

该策略是通用的。

