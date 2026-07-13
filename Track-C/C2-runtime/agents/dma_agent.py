#!/usr/bin/env python3
"""AEC-C2 DMA Agent：无状态、按通用周期模型选择策略。"""

import json
import math
import sys


def cycles(request, action):
    # 直接复用赛题给出的 DMA 虚拟周期公式，便于解释和复核。
    chunks = math.ceil(request["bytes"] / action["chunk_bytes"])
    payload = math.ceil(request["bytes"] / 32)
    parallelism = min(action["queue_depth"], request["concurrency"], 2)
    setup = 45 if action["use_zero_copy"] else 100
    alignment_penalty = 0 if request["alignment"] >= 64 else 13
    return setup + math.ceil(payload / parallelism) + 24 * (chunks - 1) + alignment_penalty


def main():
    request = json.load(sys.stdin)
    best = None
    # 合法动作空间很小，逐一枚举比写经验规则更清楚，也不会依赖 case_id。
    for chunk_bytes in (4096, 65536, 1048576):
        for queue_depth in (1, 2, 4, 8):
            for use_zero_copy in (False, True):
                if use_zero_copy and not request["registered"]:
                    continue
                action = {
                    "channel": 0,
                    "chunk_bytes": chunk_bytes,
                    "queue_depth": queue_depth,
                    "use_zero_copy": use_zero_copy,
                }
                score = cycles(request, action)
                # 同分时保持确定性：优先更小 chunk/depth，并优先允许 zero-copy 的结果。
                key = (score, chunk_bytes, queue_depth, not use_zero_copy)
                if best is None or key < best[0]:
                    best = (key, action)
    json.dump(best[1], sys.stdout, separators=(",", ":"), sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
