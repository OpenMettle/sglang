"""SWA checkpoint windows without HiCache, the way the gfx1201 serving arms
run them: device-only inserts, evictions and locks on a small SWA pool.

Upstream's randomized case (test_unified_radix_cache_checkpoint_window) needs
HiCache transfers, so it only runs on a GPU. This one keeps to the device
tier and also checks the property the DFLASH draft depends on: every cached
Mamba checkpoint still has the SWA KV of the window before it.
"""

import random
import unittest

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.checkpoint_window import (
    SWACheckpointWindowFixtures,
)
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import match_prefix
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class DeviceOnlyCheckpointWindowSuite(SWACheckpointWindowFixtures):
    def _assert_checkpoints_keep_their_windows(self, cache, seqs):
        checkpoints = 0
        for seq in seqs:
            match = match_prefix(cache, seq)
            node = match.last_device_node
            if node == cache.root_node_handle():
                continue
            spans = self._path(cache, node, match.device_prefix_len)
            for _, end, _, has_checkpoint in spans:
                if not has_checkpoint:
                    continue
                checkpoints += 1
                window_start = max(0, end - self.window)
                for start, stop, has_swa, _ in spans:
                    if stop > window_start and start < end:
                        self.assertTrue(
                            has_swa,
                            f"checkpoint at {end} lost SWA KV of [{start}, {stop})",
                        )
        return checkpoints

    def _run_random_operations(self, seed):
        # A small SWA pool, so allocations alone run it out and trigger eviction.
        cache, allocator, pool = self._build(swa_kv_size=self.cfg.kv_size // 4)
        rng = random.Random(seed)
        seqs, locks = [], []
        next_token, checked = 1, 0
        for _ in range(300):
            op = rng.random()
            if op < 0.45 or not seqs:
                base = rng.choice(seqs) if seqs and rng.random() < 0.7 else []
                keep = rng.randrange(len(base) // self.page + 1) * self.page
                extra = rng.randint(1, 3 * self.window // self.page) * self.page
                seq = base[:keep] + list(range(next_token, next_token + extra))
                next_token += extra
                if self._make_room(cache, allocator, len(seq)):
                    self._insert(cache, allocator, pool, seq)
                    seqs.append(seq)
            elif op < 0.65:
                pages = rng.randint(1, 16) * self.page
                cache.evict(
                    rng.choice(
                        [
                            EvictParams(num_tokens=pages),
                            EvictParams(num_tokens=0, window_num_tokens={"swa": pages}),
                            EvictParams(num_tokens=0, mamba_num=1),
                        ]
                    )
                )
            elif op < 0.85:
                match = match_prefix(cache, rng.choice(seqs))
                if match.last_device_node != cache.root_node_handle():
                    locks.append(cache.lock(match.last_device_node))
            elif locks:
                lock = locks.pop(rng.randrange(len(locks)))
                if not lock.swa_released and rng.random() < 0.3:
                    cache.release_swa(lock)
                    locks.append(lock)
                else:
                    cache.unlock(lock)
            cache.sanity_check()
            checked += self._assert_checkpoints_keep_their_windows(cache, seqs)
        self.assertGreater(checked, 0)
        for lock in locks:
            cache.unlock(lock)
        swa = ComponentType.SWA
        cache.evict_for_alloc(
            EvictParams(
                num_tokens=0,
                window_num_tokens={
                    "swa": cache.tree_core.component_evictable_size(swa)
                },
            )
        )
        self.assertEqual(cache.tree_core.component_evictable_size(swa), 0)
        cache.sanity_check()

    def test_random_device_operations_keep_every_checkpoint_window(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                self._run_random_operations(seed)


_CFG = dict(
    components=(ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA),
    swa_checkpoint_windows=True,
    kv_size=1024,
    max_num_reqs=400,
    mamba_cache_size=400,
)


class TestDeviceOnlyCheckpointWindowsPage1(
    DeviceOnlyCheckpointWindowSuite, CustomTestCase
):
    # no_buffer checkpoints at page size 1, as the serving arms run.
    cfg = CacheConfig(page_size=1, sliding_window_size=4, **_CFG)


class TestDeviceOnlyCheckpointWindowsPage4(
    DeviceOnlyCheckpointWindowSuite, CustomTestCase
):
    cfg = CacheConfig(
        page_size=4, sliding_window_size=6, enable_mamba_extra_buffer=True, **_CFG
    )


if __name__ == "__main__":
    unittest.main()
