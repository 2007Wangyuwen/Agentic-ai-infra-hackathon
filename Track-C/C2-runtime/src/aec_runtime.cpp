#include "aec_runtime.h"
#include "aec_device_abi.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct aecStreamOpaque {};
struct aecEventOpaque {};

namespace {

constexpr uint64_t kDeviceAlignment = 64;
constexpr uint64_t kVectorLimit = 1048576;

thread_local aecError_t last_error = AEC_SUCCESS;

struct Allocation {
    uint64_t base = 0;
    uint64_t bytes = 0;
};

struct Registration {
    uintptr_t base = 0;
    size_t bytes = 0;
};

struct Event;

struct Task {
    enum class Kind { Command, EventMarker };
    Kind kind = Kind::Command;
    aecDeviceCommand command{};
    Event *event = nullptr;
    uint64_t generation = 0;
};

struct Stream {
    uint64_t id = 0;
    std::deque<Task> queue;
    aecError_t pending_error = AEC_SUCCESS;
};

struct Event {
    uint64_t id = 0;
    bool recorded = false;
    bool completed = false;
    uint64_t generation = 0;
    uint64_t cycle = 0;
    Stream *stream = nullptr;
};

// Runtime 侧只维护 API 生命周期相关的登记表。真正的设备内存内容和 kernel 执行仍在
// 官方设备库中完成，这里只负责在生成 ABI command 前校验 handle 和地址区间。
std::mutex runtime_mutex;
std::unordered_map<uint64_t, Allocation> allocations;
std::vector<Registration> registrations;
std::unordered_set<Stream *> live_streams;
std::unordered_set<Event *> live_events;
uint64_t next_sequence = 1;
uint64_t next_stream_id = 1;
uint64_t next_event_id = 1;
uint8_t next_dma_channel = 0;

aecError_t finish(aecError_t error) {
    if (error != AEC_SUCCESS) {
        last_error = error;
    }
    return error;
}

aecError_t map_device_status(aecDeviceStatus status) {
    switch (status) {
    case AEC_DEVICE_SUCCESS:
        return AEC_SUCCESS;
    case AEC_DEVICE_INVALID_ARGUMENT:
        return AEC_ERROR_INVALID_ARGUMENT;
    case AEC_DEVICE_OUT_OF_MEMORY:
        return AEC_ERROR_OUT_OF_MEMORY;
    case AEC_DEVICE_INVALID_ADDRESS:
        return AEC_ERROR_INVALID_ADDRESS;
    case AEC_DEVICE_UNSUPPORTED:
        return AEC_ERROR_NOT_SUPPORTED;
    case AEC_DEVICE_INJECTED_FAULT:
        return AEC_ERROR_DEVICE;
    case AEC_DEVICE_ISA_TRAP:
        return AEC_ERROR_ISA_TRAP;
    case AEC_DEVICE_INTERNAL:
    default:
        return AEC_ERROR_DEVICE;
    }
}

bool add_overflows_u64(uint64_t base, uint64_t bytes) {
    return bytes > std::numeric_limits<uint64_t>::max() - base;
}

bool add_overflows_ptr(uintptr_t base, size_t bytes) {
    return bytes > std::numeric_limits<uintptr_t>::max() - base;
}

bool span_in_allocation(uint64_t ptr, uint64_t bytes) {
    if (ptr == 0 || bytes == 0 || add_overflows_u64(ptr, bytes)) {
        return false;
    }
    // C2 规范要求 span 完整落在单个 live allocation 内，不能只检查全局地址范围。
    const uint64_t end = ptr + bytes;
    for (const auto &entry : allocations) {
        const Allocation &allocation = entry.second;
        if (ptr >= allocation.base &&
            end <= allocation.base + allocation.bytes) {
            return true;
        }
    }
    return false;
}

bool host_span_registered(uintptr_t ptr, size_t bytes) {
    if (ptr == 0 || bytes == 0 || add_overflows_ptr(ptr, bytes)) {
        return false;
    }
    const uintptr_t end = ptr + bytes;
    // registered host memory 只影响 DMA flag 和虚拟周期，不改变数据正确性判断。
    for (const Registration &registration : registrations) {
        const uintptr_t reg_end = registration.base + registration.bytes;
        if (ptr >= registration.base && end <= reg_end) {
            return true;
        }
    }
    return false;
}

aecError_t build_and_submit_launch(aecKernelId kernel, aecDim3 grid,
                                   aecDim3 block, const void *args,
                                   size_t args_size,
                                   aecStream_t stream_handle);

bool registration_overlaps(uintptr_t ptr, size_t bytes) {
    if (ptr == 0 || bytes == 0 || add_overflows_ptr(ptr, bytes)) {
        return true;
    }
    const uintptr_t end = ptr + bytes;
    for (const Registration &registration : registrations) {
        const uintptr_t reg_end = registration.base + registration.bytes;
        if (ptr < reg_end && registration.base < end) {
            return true;
        }
    }
    return false;
}

Stream *as_stream(aecStream_t stream) {
    return reinterpret_cast<Stream *>(stream);
}

Event *as_event(aecEvent_t event) {
    return reinterpret_cast<Event *>(event);
}

bool is_live_stream(Stream *stream) {
    return stream != nullptr && live_streams.find(stream) != live_streams.end();
}

bool is_live_event(Event *event) {
    return event != nullptr && live_events.find(event) != live_events.end();
}

uint32_t block_volume(aecDim3 block) {
    if (block.x == 0 || block.y == 0 || block.z == 0) {
        return 0;
    }
    const uint64_t volume = static_cast<uint64_t>(block.x) * block.y * block.z;
    return volume > std::numeric_limits<uint32_t>::max()
               ? std::numeric_limits<uint32_t>::max()
               : static_cast<uint32_t>(volume);
}

void put_le32(uint8_t *buffer, size_t offset, uint32_t value) {
    buffer[offset + 0] = static_cast<uint8_t>(value);
    buffer[offset + 1] = static_cast<uint8_t>(value >> 8);
    buffer[offset + 2] = static_cast<uint8_t>(value >> 16);
    buffer[offset + 3] = static_cast<uint8_t>(value >> 24);
}

void put_le64(uint8_t *buffer, size_t offset, uint64_t value) {
    for (size_t index = 0; index < 8; ++index) {
        buffer[offset + index] = static_cast<uint8_t>(value >> (index * 8));
    }
}

void put_float_bits(uint8_t *buffer, size_t offset, float value) {
    // 参数块要求 little-endian 紧密布局；float 先按位取出再写入，避免 native struct padding。
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit");
    std::memcpy(&bits, &value, sizeof(bits));
    put_le32(buffer, offset, bits);
}

uint64_t ceil_div(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

uint64_t dtype_input_bytes(uint32_t dtype, uint64_t elements) {
    switch (dtype) {
    case AEC_DTYPE_FP4_E2M1:
    case AEC_DTYPE_INT4:
        return ceil_div(elements, 2);
    case AEC_DTYPE_FP8_E4M3:
    case AEC_DTYPE_FP8_E5M2:
    case AEC_DTYPE_INT8:
        return elements;
    case AEC_DTYPE_FP16:
    case AEC_DTYPE_BF16:
        return elements * 2;
    case AEC_DTYPE_FP32:
    case AEC_DTYPE_INT32:
        return elements * 4;
    case AEC_DTYPE_FP64:
        return elements * 8;
    default:
        return 0;
    }
}

uint64_t dtype_output_bytes(uint32_t dtype, uint64_t elements) {
    switch (dtype) {
    case AEC_DTYPE_INT4:
    case AEC_DTYPE_INT8:
    case AEC_DTYPE_INT32:
        return elements * 4;
    default:
        return dtype_input_bytes(dtype, elements);
    }
}

// 将公开 Runtime 调用转换为 canonical device command。aecLaunch 和计算库 wrapper 都走
// 这条路径，因此成功计算一定会 resolve 官方 image 并提交给受控设备，而不是在 Host 侧
// 直接算出答案。
struct LaunchBuild {
    aecDeviceCommand command{};
    aecError_t error = AEC_SUCCESS;
};

uint32_t variant_for_kernel(aecKernelId kernel) {
    switch (kernel) {
    case AEC_KERNEL_GEMM_NAIVE:
        return AEC_KERNEL_VARIANT_NAIVE;
    case AEC_KERNEL_GEMM_TILED:
        return AEC_KERNEL_VARIANT_TILED;
    case AEC_KERNEL_GEMM_VECTORIZED:
        return AEC_KERNEL_VARIANT_VECTORIZED;
    default:
        return AEC_KERNEL_VARIANT_DEFAULT;
    }
}

uint32_t workspace_for_variant(uint32_t variant) {
    // workspace 数值与 manifest 中三个 GEMM variant 的约束保持一致。
    if (variant == AEC_KERNEL_VARIANT_TILED) return 4096;
    if (variant == AEC_KERNEL_VARIANT_VECTORIZED) return 8192;
    return 0;
}

aecDim3 default_grid_for_count(uint64_t count) {
    return aecDim3{static_cast<uint32_t>(ceil_div(count, 256)), 1, 1};
}

aecDim3 default_block_for_count(uint64_t) {
    return aecDim3{256, 1, 1};
}

aecDim3 single_grid() {
    return aecDim3{1, 1, 1};
}

aecDim3 single_block() {
    return aecDim3{1, 1, 1};
}

aecDeviceDim3 to_device_dim(aecDim3 dim) {
    return aecDeviceDim3{dim.x, dim.y, dim.z};
}

LaunchBuild build_launch(aecKernelId kernel, aecDim3 grid, aecDim3 block,
                         const void *args, size_t args_size,
                         uint64_t stream_id) {
    LaunchBuild build{};
    if (grid.x == 0 || grid.y == 0 || grid.z == 0 ||
        block_volume(block) == 0 || block_volume(block) > 1024 ||
        args == nullptr) {
        build.error = AEC_ERROR_INVALID_ARGUMENT;
        return build;
    }

    uint32_t semantic_kernel = static_cast<uint32_t>(kernel);
    uint32_t dtype = AEC_DTYPE_FP32;
    uint32_t variant = variant_for_kernel(kernel);
    uint32_t parameter_bytes = 0;
    uint8_t parameters[AEC_DEVICE_MAX_PARAM_BYTES]{};

    if (kernel == AEC_KERNEL_VECTOR_ADD_F32) {
        if (args_size != sizeof(aecVectorAddArgs)) {
            build.error = AEC_ERROR_INVALID_ARGUMENT;
            return build;
        }
        const auto *values = static_cast<const aecVectorAddArgs *>(args);
        if (values->count == 0 || values->count > kVectorLimit) {
            build.error = AEC_ERROR_INVALID_ARGUMENT;
            return build;
        }
        parameter_bytes = AEC_KERNEL_PARAM_VECTOR_ADD_BYTES;
        put_le64(parameters, 0, values->a);
        put_le64(parameters, 8, values->b);
        put_le64(parameters, 16, values->c);
        put_le64(parameters, 24, values->count);
    } else if (kernel == AEC_KERNEL_GEMM_NAIVE ||
               kernel == AEC_KERNEL_GEMM_TILED ||
               kernel == AEC_KERNEL_GEMM_VECTORIZED) {
        // GEMM 的 dtype 来自 Runtime API 参数；不要把 Runtime enum 和 ISA selector 混用。
        if (args_size != sizeof(aecGemmArgs)) {
            build.error = AEC_ERROR_INVALID_ARGUMENT;
            return build;
        }
        const auto *values = static_cast<const aecGemmArgs *>(args);
        if (values->m == 0 || values->n == 0 || values->k == 0 ||
            values->m > 256 || values->n > 256 || values->k > 256 ||
            dtype_input_bytes(values->dtype, 1) == 0) {
            build.error = AEC_ERROR_INVALID_ARGUMENT;
            return build;
        }
        dtype = values->dtype;
        parameter_bytes = AEC_KERNEL_PARAM_GEMM_BYTES;
        put_le64(parameters, 0, values->a);
        put_le64(parameters, 8, values->b);
        put_le64(parameters, 16, values->c);
        put_le32(parameters, 24, values->m);
        put_le32(parameters, 28, values->n);
        put_le32(parameters, 32, values->k);
        put_le32(parameters, 36, values->dtype);
    } else if (kernel == AEC_KERNEL_AXPY_F32) {
        if (args_size != sizeof(aecAxpyArgs)) {
            build.error = AEC_ERROR_INVALID_ARGUMENT;
            return build;
        }
        const auto *values = static_cast<const aecAxpyArgs *>(args);
        if (values->count == 0 || values->count > kVectorLimit) {
            build.error = AEC_ERROR_INVALID_ARGUMENT;
            return build;
        }
        parameter_bytes = AEC_KERNEL_PARAM_AXPY_BYTES;
        put_le64(parameters, 0, values->x);
        put_le64(parameters, 8, values->y);
        put_le64(parameters, 16, values->count);
        put_float_bits(parameters, 24, values->alpha);
    } else if (kernel == AEC_KERNEL_DOT_F32) {
        if (args_size != sizeof(aecDotArgs)) {
            build.error = AEC_ERROR_INVALID_ARGUMENT;
            return build;
        }
        const auto *values = static_cast<const aecDotArgs *>(args);
        if (values->count == 0 || values->count > kVectorLimit) {
            build.error = AEC_ERROR_INVALID_ARGUMENT;
            return build;
        }
        parameter_bytes = AEC_KERNEL_PARAM_DOT_BYTES;
        put_le64(parameters, 0, values->x);
        put_le64(parameters, 8, values->y);
        put_le64(parameters, 16, values->result);
        put_le64(parameters, 24, values->count);
    } else if (kernel == AEC_KERNEL_NRM2_F32) {
        if (args_size != sizeof(aecNrm2Args)) {
            build.error = AEC_ERROR_INVALID_ARGUMENT;
            return build;
        }
        const auto *values = static_cast<const aecNrm2Args *>(args);
        if (values->count == 0 || values->count > kVectorLimit) {
            build.error = AEC_ERROR_INVALID_ARGUMENT;
            return build;
        }
        parameter_bytes = AEC_KERNEL_PARAM_NRM2_BYTES;
        put_le64(parameters, 0, values->x);
        put_le64(parameters, 8, values->result);
        put_le64(parameters, 16, values->count);
    } else {
        build.error = AEC_ERROR_NOT_SUPPORTED;
        return build;
    }

    aecDeviceKernelInfo info{};
    const aecDeviceStatus status = aecDeviceResolveKernel(
        semantic_kernel, dtype, variant, &info);
    if (status != AEC_DEVICE_SUCCESS) {
        build.error = map_device_status(status);
        return build;
    }
    if (info.parameter_bytes != parameter_bytes) {
        build.error = AEC_ERROR_DEVICE;
        return build;
    }

    aecDeviceCommand command{};
    command.abi_version = AEC_DEVICE_ABI_VERSION;
    command.opcode = AEC_DEVICE_OP_ISA_LAUNCH;
    command.sequence = 0;
    command.stream_id = stream_id;
    command.kernel_handle = info.handle;
    command.isa_version = info.isa_version;
    command.entry_pc = info.entry_pc;
    command.grid = to_device_dim(grid);
    command.block = to_device_dim(block);
    command.parameter_bytes = parameter_bytes;
    command.dynamic_shared_bytes = workspace_for_variant(variant);
    std::memcpy(command.parameters, parameters, parameter_bytes);
    build.command = command;
    return build;
}

aecError_t execute_command(aecDeviceCommand command) {
    command.sequence = next_sequence++;
    aecDeviceCompletion completion{};
    const aecDeviceStatus status = aecDeviceSubmit(&command, &completion);
    if (status != AEC_DEVICE_SUCCESS) {
        return map_device_status(status);
    }
    if (completion.status != AEC_DEVICE_SUCCESS) {
        return map_device_status(static_cast<aecDeviceStatus>(completion.status));
    }
    if (completion.sequence != command.sequence) {
        return AEC_ERROR_DEVICE;
    }
    return AEC_SUCCESS;
}

aecError_t execute_dma(aecDeviceCommand command) {
    const bool h2d = command.opcode == AEC_DEVICE_OP_H2D;
    const uint64_t device_ptr = h2d ? command.dst : command.src;
    if (command.host_address == 0 || command.bytes == 0) {
        return AEC_ERROR_INVALID_ARGUMENT;
    }
    if (!span_in_allocation(device_ptr, command.bytes)) {
        return AEC_ERROR_INVALID_ADDRESS;
    }
    // 注册内存只改变 device command flag 和虚拟周期模型，数据搬运仍提交同一套 ABI。
    const bool registered =
        host_span_registered(static_cast<uintptr_t>(command.host_address),
                             static_cast<size_t>(command.bytes));
    if (registered) {
        command.flags |= AEC_DEVICE_FLAG_REGISTERED | AEC_DEVICE_FLAG_ZERO_COPY;
    }
    command.channel = next_dma_channel;
    next_dma_channel = static_cast<uint8_t>((next_dma_channel + 1) % 2);
    command.chunk_bytes = command.bytes <= 65536 ? 65536 : 1048576;
    command.queue_depth = 2;
    return execute_command(command);
}

uint64_t current_total_cycles() {
    aecDeviceStats stats{};
    if (aecDeviceGetStats(&stats) != AEC_DEVICE_SUCCESS) {
        return 0;
    }
    return stats.total_virtual_cycles;
}

aecError_t drain_stream(Stream *stream, Event *until_event = nullptr,
                        uint64_t until_generation = 0) {
    // 虚拟设备是确定性的，所以 stream 在同步点惰性执行即可；关键是保持同一 stream FIFO。
    while (!stream->queue.empty()) {
        Task task = stream->queue.front();
        stream->queue.pop_front();
        if (task.kind == Task::Kind::EventMarker) {
            if (is_live_event(task.event) &&
                task.event->generation == task.generation) {
                task.event->completed = true;
                task.event->cycle = current_total_cycles();
            }
            if (task.event == until_event && task.generation == until_generation) {
                return AEC_SUCCESS;
            }
            continue;
        }

        aecError_t error = AEC_SUCCESS;
        if (task.command.opcode == AEC_DEVICE_OP_H2D ||
            task.command.opcode == AEC_DEVICE_OP_D2H) {
            error = execute_dma(task.command);
        } else {
            error = execute_command(task.command);
        }
        if (error != AEC_SUCCESS) {
            stream->pending_error = error;
            return error;
        }
    }
    return AEC_SUCCESS;
}

void drain_all_streams_ignore_errors() {
    for (Stream *stream : live_streams) {
        if (stream != nullptr) {
            (void)drain_stream(stream);
            stream->pending_error = AEC_SUCCESS;
        }
    }
}

aecError_t consume_stream_error(Stream *stream, aecError_t drain_result) {
    if (stream->pending_error != AEC_SUCCESS) {
        const aecError_t error = stream->pending_error;
        stream->pending_error = AEC_SUCCESS;
        return error;
    }
    return drain_result;
}

aecError_t submit_or_enqueue(aecDeviceCommand command, Stream *stream) {
    if (stream == nullptr) {
        if (command.opcode == AEC_DEVICE_OP_H2D ||
            command.opcode == AEC_DEVICE_OP_D2H) {
            return execute_dma(command);
        }
        return execute_command(command);
    }
    Task task{};
    task.kind = Task::Kind::Command;
    task.command = command;
    stream->queue.push_back(task);
    return AEC_SUCCESS;
}

aecDeviceCommand make_dma_command(aecDeviceOpcode opcode, aecDevicePtr device_ptr,
                                  void *host_ptr, size_t bytes,
                                  uint64_t stream_id) {
    aecDeviceCommand command{};
    command.abi_version = AEC_DEVICE_ABI_VERSION;
    command.opcode = opcode;
    command.sequence = 0;
    command.stream_id = stream_id;
    command.host_address = reinterpret_cast<uintptr_t>(host_ptr);
    command.bytes = bytes;
    if (opcode == AEC_DEVICE_OP_H2D) {
        command.dst = device_ptr;
    } else {
        command.src = device_ptr;
    }
    return command;
}

uint32_t choose_gemm_kernel(uint32_t m, uint32_t n, uint32_t k,
                            aecDevicePtr a, aecDevicePtr b, aecDevicePtr c) {
    // 这里按通用 shape/alignment 约束选择 image，不依赖具体测试输入。
    const bool aligned16 = (a % 16 == 0) && (b % 16 == 0) && (c % 16 == 0);
    if (aligned16 && m % 8 == 0 && n % 8 == 0 && k % 8 == 0) {
        return AEC_KERNEL_GEMM_VECTORIZED;
    }
    if (m % 4 == 0 && n % 4 == 0 && k % 4 == 0) {
        return AEC_KERNEL_GEMM_TILED;
    }
    return AEC_KERNEL_GEMM_NAIVE;
}

aecError_t launch_gemm(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                       uint32_t m, uint32_t n, uint32_t k, uint32_t dtype,
                       aecStream_t stream) {
    if (m == 0 || n == 0 || k == 0 || m > 256 || n > 256 || k > 256 ||
        dtype_input_bytes(dtype, 1) == 0) {
        return finish(AEC_ERROR_INVALID_ARGUMENT);
    }
    (void)dtype_output_bytes(dtype, static_cast<uint64_t>(m) * n);
    const uint32_t kernel = choose_gemm_kernel(m, n, k, a, b, c);
    const aecGemmArgs args{a, b, c, m, n, k, dtype, 0};
    return finish(build_and_submit_launch(static_cast<aecKernelId>(kernel),
                                          single_grid(), single_block(), &args,
                                          sizeof(args), stream));
}

aecError_t build_and_submit_launch(aecKernelId kernel, aecDim3 grid,
                                   aecDim3 block, const void *args,
                                   size_t args_size, aecStream_t stream_handle) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    Stream *stream = nullptr;
    if (stream_handle != nullptr) {
        stream = as_stream(stream_handle);
        if (!is_live_stream(stream)) {
            return AEC_ERROR_INVALID_HANDLE;
        }
    }
    const uint64_t stream_id = stream == nullptr ? 0 : stream->id;
    LaunchBuild build = build_launch(kernel, grid, block, args, args_size, stream_id);
    if (build.error != AEC_SUCCESS) {
        return build.error;
    }
    return submit_or_enqueue(build.command, stream);
}

} // namespace

extern "C" {

aecError_t aecDeviceCount(int *count) {
    if (count == nullptr) return finish(AEC_ERROR_INVALID_ARGUMENT);
    aecDeviceCaps caps{};
    if (aecDeviceGetCaps(&caps) != AEC_DEVICE_SUCCESS) return finish(AEC_ERROR_DEVICE);
    *count = static_cast<int>(caps.device_count);
    return AEC_SUCCESS;
}

aecError_t aecDeviceInfo(int device, aecDeviceInfoData *info) {
    if (device != 0 || info == nullptr) return finish(AEC_ERROR_INVALID_ARGUMENT);
    aecDeviceCaps caps{};
    if (aecDeviceGetCaps(&caps) != AEC_DEVICE_SUCCESS) return finish(AEC_ERROR_DEVICE);
    *info = {};
    info->abi_version = AEC_RUNTIME_ABI_VERSION;
    std::strncpy(info->name, "AEC Deterministic Virtual Device", sizeof(info->name) - 1);
    info->memory_bytes = caps.memory_bytes;
    info->dma_channels = caps.dma_channels;
    info->max_threads_per_block = caps.max_threads_per_block;
    info->isa_version = caps.isa_version;
    info->isa_profile = caps.isa_profile;
    info->max_parameter_bytes = caps.max_parameter_bytes;
    return AEC_SUCCESS;
}

aecError_t aecGetLastError(void) {
    const aecError_t value = last_error;
    last_error = AEC_SUCCESS;
    return value;
}

aecError_t aecPeekAtLastError(void) { return last_error; }

const char *aecGetErrorName(aecError_t error) {
    switch (error) {
    case AEC_SUCCESS: return "AEC_SUCCESS";
    case AEC_ERROR_INVALID_ARGUMENT: return "AEC_ERROR_INVALID_ARGUMENT";
    case AEC_ERROR_OUT_OF_MEMORY: return "AEC_ERROR_OUT_OF_MEMORY";
    case AEC_ERROR_INVALID_HANDLE: return "AEC_ERROR_INVALID_HANDLE";
    case AEC_ERROR_INVALID_ADDRESS: return "AEC_ERROR_INVALID_ADDRESS";
    case AEC_ERROR_NOT_READY: return "AEC_ERROR_NOT_READY";
    case AEC_ERROR_NOT_SUPPORTED: return "AEC_ERROR_NOT_SUPPORTED";
    case AEC_ERROR_DEVICE: return "AEC_ERROR_DEVICE";
    case AEC_ERROR_INTERNAL: return "AEC_ERROR_INTERNAL";
    case AEC_ERROR_ISA_TRAP: return "AEC_ERROR_ISA_TRAP";
    default: return "AEC_ERROR_UNKNOWN";
    }
}

aecError_t aecAlloc(aecDevicePtr *out_ptr, size_t bytes) {
    if (out_ptr == nullptr || bytes == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
    std::lock_guard<std::mutex> lock(runtime_mutex);
    aecDevicePtr ptr = 0;
    const aecDeviceStatus status = aecDeviceAlloc(bytes, kDeviceAlignment, &ptr);
    if (status != AEC_DEVICE_SUCCESS) return finish(map_device_status(status));
    allocations[ptr] = Allocation{ptr, static_cast<uint64_t>(bytes)};
    *out_ptr = ptr;
    return AEC_SUCCESS;
}

aecError_t aecFree(aecDevicePtr ptr) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    auto it = allocations.find(ptr);
    if (ptr == 0 || it == allocations.end()) {
        return finish(AEC_ERROR_INVALID_ADDRESS);
    }
    drain_all_streams_ignore_errors();
    const aecDeviceStatus status = aecDeviceFree(ptr);
    if (status != AEC_DEVICE_SUCCESS) return finish(map_device_status(status));
    allocations.erase(it);
    return AEC_SUCCESS;
}

aecError_t aecCopyH2D(aecDevicePtr dst, const void *src, size_t bytes) {
    if (src == nullptr || bytes == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
    std::lock_guard<std::mutex> lock(runtime_mutex);
    if (!span_in_allocation(dst, bytes)) return finish(AEC_ERROR_INVALID_ADDRESS);
    aecDeviceCommand command = make_dma_command(AEC_DEVICE_OP_H2D, dst,
                                                const_cast<void *>(src), bytes, 0);
    return finish(execute_dma(command));
}

aecError_t aecCopyD2H(void *dst, aecDevicePtr src, size_t bytes) {
    if (dst == nullptr || bytes == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
    std::lock_guard<std::mutex> lock(runtime_mutex);
    if (!span_in_allocation(src, bytes)) return finish(AEC_ERROR_INVALID_ADDRESS);
    aecDeviceCommand command = make_dma_command(AEC_DEVICE_OP_D2H, src, dst, bytes, 0);
    return finish(execute_dma(command));
}

aecError_t aecCopyAsync(aecDevicePtr device_ptr, void *host_ptr, size_t bytes,
                        aecCopyDirection direction, aecStream_t stream_handle) {
    if (host_ptr == nullptr || bytes == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
    if (direction != AEC_COPY_HOST_TO_DEVICE &&
        direction != AEC_COPY_DEVICE_TO_HOST) {
        return finish(AEC_ERROR_INVALID_ARGUMENT);
    }
    std::lock_guard<std::mutex> lock(runtime_mutex);
    Stream *stream = as_stream(stream_handle);
    if (!is_live_stream(stream)) return finish(AEC_ERROR_INVALID_HANDLE);
    const aecDeviceOpcode opcode = direction == AEC_COPY_HOST_TO_DEVICE
                                       ? AEC_DEVICE_OP_H2D
                                       : AEC_DEVICE_OP_D2H;
    aecDeviceCommand command = make_dma_command(opcode, device_ptr, host_ptr,
                                                bytes, stream->id);
    Task task{};
    task.kind = Task::Kind::Command;
    task.command = command;
    stream->queue.push_back(task);
    return AEC_SUCCESS;
}

aecError_t aecStreamCreate(aecStream_t *out_stream) {
    if (out_stream == nullptr) return finish(AEC_ERROR_INVALID_ARGUMENT);
    std::lock_guard<std::mutex> lock(runtime_mutex);
    Stream *stream = new Stream();
    stream->id = next_stream_id++;
    live_streams.insert(stream);
    *out_stream = reinterpret_cast<aecStream_t>(stream);
    return AEC_SUCCESS;
}

aecError_t aecStreamDestroy(aecStream_t stream_handle) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    Stream *stream = as_stream(stream_handle);
    if (!is_live_stream(stream)) return finish(AEC_ERROR_INVALID_HANDLE);
    // 先移出 live registry，再 drain 队列；后续重复使用同一 handle 会被判为 invalid。
    live_streams.erase(stream);
    (void)drain_stream(stream);
    delete stream;
    return AEC_SUCCESS;
}

aecError_t aecStreamSync(aecStream_t stream_handle) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    Stream *stream = as_stream(stream_handle);
    if (!is_live_stream(stream)) return finish(AEC_ERROR_INVALID_HANDLE);
    const aecError_t result = drain_stream(stream);
    return finish(consume_stream_error(stream, result));
}

aecError_t aecEventCreate(aecEvent_t *out_event) {
    if (out_event == nullptr) return finish(AEC_ERROR_INVALID_ARGUMENT);
    std::lock_guard<std::mutex> lock(runtime_mutex);
    Event *event = new Event();
    event->id = next_event_id++;
    live_events.insert(event);
    *out_event = reinterpret_cast<aecEvent_t>(event);
    return AEC_SUCCESS;
}

aecError_t aecEventDestroy(aecEvent_t event_handle) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    Event *event = as_event(event_handle);
    if (!is_live_event(event)) return finish(AEC_ERROR_INVALID_HANDLE);
    if (event->recorded && !event->completed && is_live_stream(event->stream)) {
        (void)drain_stream(event->stream, event, event->generation);
    }
    live_events.erase(event);
    delete event;
    return AEC_SUCCESS;
}

aecError_t aecEventRecord(aecEvent_t event_handle, aecStream_t stream_handle) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    Event *event = as_event(event_handle);
    Stream *stream = as_stream(stream_handle);
    if (!is_live_event(event) || !is_live_stream(stream)) {
        return finish(AEC_ERROR_INVALID_HANDLE);
    }
    event->recorded = true;
    event->completed = false;
    event->stream = stream;
    // 重复 record 使用 generation 区分新旧 marker，避免 query 读到旧事件结果。
    event->generation++;
    Task marker{};
    marker.kind = Task::Kind::EventMarker;
    marker.event = event;
    marker.generation = event->generation;
    stream->queue.push_back(marker);
    return AEC_SUCCESS;
}

aecError_t aecEventSynchronize(aecEvent_t event_handle) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    Event *event = as_event(event_handle);
    if (!is_live_event(event)) return finish(AEC_ERROR_INVALID_HANDLE);
    if (!event->recorded) return finish(AEC_ERROR_INVALID_ARGUMENT);
    if (!event->completed && is_live_stream(event->stream)) {
        const aecError_t result =
            drain_stream(event->stream, event, event->generation);
        if (result != AEC_SUCCESS) return finish(result);
    }
    return event->completed ? AEC_SUCCESS : finish(AEC_ERROR_NOT_READY);
}

aecError_t aecEventQuery(aecEvent_t event_handle) {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    Event *event = as_event(event_handle);
    if (!is_live_event(event)) return finish(AEC_ERROR_INVALID_HANDLE);
    if (!event->recorded) return finish(AEC_ERROR_INVALID_ARGUMENT);
    return event->completed ? AEC_SUCCESS : finish(AEC_ERROR_NOT_READY);
}

aecError_t aecEventElapsedCycles(aecEvent_t start_handle, aecEvent_t end_handle,
                                 uint64_t *cycles) {
    if (cycles == nullptr) return finish(AEC_ERROR_INVALID_ARGUMENT);
    std::lock_guard<std::mutex> lock(runtime_mutex);
    Event *start = as_event(start_handle);
    Event *end = as_event(end_handle);
    if (!is_live_event(start) || !is_live_event(end)) {
        return finish(AEC_ERROR_INVALID_HANDLE);
    }
    if (!start->recorded || !end->recorded || !start->completed ||
        !end->completed || end->cycle < start->cycle) {
        return finish(AEC_ERROR_INVALID_ARGUMENT);
    }
    *cycles = end->cycle - start->cycle;
    return AEC_SUCCESS;
}

aecError_t aecHostRegister(void *ptr, size_t bytes) {
    if (ptr == nullptr || bytes == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
    std::lock_guard<std::mutex> lock(runtime_mutex);
    const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
    if (registration_overlaps(base, bytes)) return finish(AEC_ERROR_INVALID_ARGUMENT);
    registrations.push_back(Registration{base, bytes});
    return AEC_SUCCESS;
}

aecError_t aecHostUnregister(void *ptr) {
    if (ptr == nullptr) return finish(AEC_ERROR_INVALID_ARGUMENT);
    std::lock_guard<std::mutex> lock(runtime_mutex);
    const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
    auto it = std::find_if(registrations.begin(), registrations.end(),
                           [base](const Registration &registration) {
                               return registration.base == base;
                           });
    if (it == registrations.end()) return finish(AEC_ERROR_INVALID_ARGUMENT);
    drain_all_streams_ignore_errors();
    registrations.erase(it);
    return AEC_SUCCESS;
}

aecError_t aecGetRuntimeStats(aecRuntimeStats *stats) {
    if (stats == nullptr) return finish(AEC_ERROR_INVALID_ARGUMENT);
    aecDeviceStats device_stats{};
    if (aecDeviceGetStats(&device_stats) != AEC_DEVICE_SUCCESS) return finish(AEC_ERROR_DEVICE);
    static_assert(sizeof(*stats) == sizeof(device_stats),
                  "runtime and device stats must match");
    std::memcpy(stats, &device_stats, sizeof(*stats));
    stats->abi_version = AEC_RUNTIME_ABI_VERSION;
    return AEC_SUCCESS;
}

aecError_t aecResetRuntimeStats(void) {
    return aecDeviceResetStats() == AEC_DEVICE_SUCCESS ? AEC_SUCCESS
                                                        : finish(AEC_ERROR_DEVICE);
}

aecError_t aecLaunch(aecKernelId kernel, aecDim3 grid, aecDim3 block,
                     const void *args, size_t args_size, aecStream_t stream) {
    return finish(build_and_submit_launch(kernel, grid, block, args, args_size, stream));
}

aecError_t aecMatmulF4(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                       uint32_t m, uint32_t n, uint32_t k,
                       aecStream_t stream) {
    return launch_gemm(a, b, c, m, n, k, AEC_DTYPE_FP4_E2M1, stream);
}

aecError_t aecMatmulF8(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                       uint32_t m, uint32_t n, uint32_t k,
                       aecFp8Format format, aecStream_t stream) {
    if (format == AEC_FP8_E4M3) {
        return launch_gemm(a, b, c, m, n, k, AEC_DTYPE_FP8_E4M3, stream);
    }
    if (format == AEC_FP8_E5M2) {
        return launch_gemm(a, b, c, m, n, k, AEC_DTYPE_FP8_E5M2, stream);
    }
    return finish(AEC_ERROR_INVALID_ARGUMENT);
}

aecError_t aecMatmulF16(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k,
                        aecStream_t stream) {
    return launch_gemm(a, b, c, m, n, k, AEC_DTYPE_FP16, stream);
}

aecError_t aecMatmulBF16(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                         uint32_t m, uint32_t n, uint32_t k,
                         aecStream_t stream) {
    return launch_gemm(a, b, c, m, n, k, AEC_DTYPE_BF16, stream);
}

aecError_t aecMatmulF32(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k,
                        aecStream_t stream) {
    return launch_gemm(a, b, c, m, n, k, AEC_DTYPE_FP32, stream);
}

aecError_t aecMatmulF64(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k,
                        aecStream_t stream) {
    return launch_gemm(a, b, c, m, n, k, AEC_DTYPE_FP64, stream);
}

aecError_t aecMatmulI4(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                       uint32_t m, uint32_t n, uint32_t k,
                       aecStream_t stream) {
    return launch_gemm(a, b, c, m, n, k, AEC_DTYPE_INT4, stream);
}

aecError_t aecMatmulI8(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                       uint32_t m, uint32_t n, uint32_t k,
                       aecStream_t stream) {
    return launch_gemm(a, b, c, m, n, k, AEC_DTYPE_INT8, stream);
}

aecError_t aecMatmulI32(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k,
                        aecStream_t stream) {
    return launch_gemm(a, b, c, m, n, k, AEC_DTYPE_INT32, stream);
}

aecError_t aecAxpy(aecDevicePtr x, aecDevicePtr y, uint64_t count, float alpha,
                   aecStream_t stream) {
    if (count == 0 || count > kVectorLimit) return finish(AEC_ERROR_INVALID_ARGUMENT);
    const aecAxpyArgs args{x, y, count, alpha, 0};
    return finish(build_and_submit_launch(AEC_KERNEL_AXPY_F32,
                                          default_grid_for_count(count),
                                          default_block_for_count(count), &args,
                                          sizeof(args), stream));
}

aecError_t aecDot(aecDevicePtr x, aecDevicePtr y, aecDevicePtr result,
                  uint64_t count, aecStream_t stream) {
    if (count == 0 || count > kVectorLimit) return finish(AEC_ERROR_INVALID_ARGUMENT);
    const aecDotArgs args{x, y, result, count};
    return finish(build_and_submit_launch(AEC_KERNEL_DOT_F32, single_grid(),
                                          single_block(), &args, sizeof(args),
                                          stream));
}

aecError_t aecNrm2(aecDevicePtr x, aecDevicePtr result, uint64_t count,
                   aecStream_t stream) {
    if (count == 0 || count > kVectorLimit) return finish(AEC_ERROR_INVALID_ARGUMENT);
    const aecNrm2Args args{x, result, count};
    return finish(build_and_submit_launch(AEC_KERNEL_NRM2_F32, single_grid(),
                                          single_block(), &args, sizeof(args),
                                          stream));
}

} // extern "C"
