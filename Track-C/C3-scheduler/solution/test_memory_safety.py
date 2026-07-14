#!/usr/bin/env python3
"""Memory pool safety boundary tests — C3.4 code-review requirement."""

from scheduler.memory import DeviceMemoryPool, MemoryBlock


def expect_exception(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


ok = 0
total = 0


def check(name: str, passed: bool, detail: str = "") -> None:
    global ok, total
    total += 1
    tag = "PASS" if passed else "FAIL"
    d = f"  {detail}" if detail else ""
    print(f"  [{tag}] {name}{d}")
    if passed:
        ok += 1


# ═══════════════════════════════════════════════════════════════
# Test 1: mallloc(0) rejected
# ═══════════════════════════════════════════════════════════════
pool = DeviceMemoryPool()
expect_exception(ValueError, lambda: pool.malloc(0))
check("malloc(0) rejected", True)

# ═══════════════════════════════════════════════════════════════
# Test 2: malloc(-1) rejected
# ═══════════════════════════════════════════════════════════════
pool = DeviceMemoryPool()
expect_exception(ValueError, lambda: pool.malloc(-1))
check("malloc(-1) rejected", True)

# ═══════════════════════════════════════════════════════════════
# Test 3: malloc(-300) rejected
# ═══════════════════════════════════════════════════════════════
pool = DeviceMemoryPool()
expect_exception(ValueError, lambda: pool.malloc(-300))
check("malloc(-300) rejected", True)

# ═══════════════════════════════════════════════════════════════
# Test 4: double-free rejected
# ═══════════════════════════════════════════════════════════════
pool = DeviceMemoryPool()
b = pool.malloc(256)
pool.free(b)
expect_exception(ValueError, lambda: pool.free(b))
check("double-free rejected", True)

# ═══════════════════════════════════════════════════════════════
# Test 5: N01 — cross-pool same-value block rejected (core fix)
# ═══════════════════════════════════════════════════════════════
pool_a = DeviceMemoryPool()
pool_b = DeviceMemoryPool()
block_a = pool_a.malloc(256)
block_b = pool_b.malloc(256)
# dataclass __eq__ is value-based: same fields → equal
assert block_a == block_b, "same-size blocks should be == by value"
# identity differs — critical for the fix
assert block_a is not block_b, "blocks from different pools must not be is"
# free(block_b) into pool_a must be rejected
expect_exception(ValueError, lambda: pool_a.free(block_b))
# external block must not be modified by the rejected free
assert block_b.free is False, "external block state must be unchanged"
assert block_a.free is False, "owned block state must be unchanged"
check("cross-pool same-value free rejected", True)
check("external block not mutated", block_b.free is False)
check("owned block not mutated", block_a.free is False)
# normal release after cross-pool check
pool_a.free(block_a)
assert block_a.free is True
check("normal release succeeds", True)
# double-free after cross-pool
expect_exception(ValueError, lambda: pool_a.free(block_a))
check("double-free after cross-pool rejected", True)
# both pools must have non-negative capacity
assert pool_a.capacity >= 0
assert pool_b.capacity >= 0
check("both pools capacity >= 0", pool_a.capacity >= 0 and pool_b.capacity >= 0,
      f"pool_a={pool_a.capacity} pool_b={pool_b.capacity}")

# ═══════════════════════════════════════════════════════════════
# Test 6: external block rejected (no-pool block)
# ═══════════════════════════════════════════════════════════════
pool = DeviceMemoryPool()
ext = MemoryBlock(0, 256, False)
expect_exception(ValueError, lambda: pool.free(ext))
check("external block rejected", True)

# ═══════════════════════════════════════════════════════════════
# Test 6: capacity stays non‑negative
# ═══════════════════════════════════════════════════════════════
pool = DeviceMemoryPool()
try:
    pool.malloc(0)
except ValueError:
    pass
check("capacity >= 0 after rejected alloc", pool.capacity >= 0,
      f"capacity={pool.capacity}")
try:
    pool.malloc(-300)
except ValueError:
    pass
check("capacity >= 0 after rejected negative alloc", pool.capacity >= 0,
      f"capacity={pool.capacity}")

# ═══════════════════════════════════════════════════════════════
# Test 7: happy-path reuse after free
# ═══════════════════════════════════════════════════════════════
pool = DeviceMemoryPool()
a = pool.malloc(256)
b = pool.malloc(512)
pool.free(a)
c = pool.malloc(128)
# c should reuse a's old slot (offset 0)
reused_ok = (c.offset == 0) and (c.size >= 128)
# b's slot (offset 256) should still be allocated
b_still_alloc = not b.free
check("reuse after free (offset)", reused_ok,
      f"a.offset=0 a.freed=True  c.offset={c.offset} c.size={c.size}")
check("other block still allocated", b_still_alloc,
      f"b.free={b.free}")

# ── summary ──
print(f"\nSummary: {ok}/{total} passed")
if ok == total:
    print("ALL PASSED!")
else:
    raise SystemExit(1)
