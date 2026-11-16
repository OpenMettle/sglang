"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    # Elastic tail (elastic_kv_tail.py), off unless enable_elastic_tail() is
    # called: slots above _tail_base are handed out only while the tail is
    # open, and only once no slot at or below it is free.
    _tail_base: int | None = None
    _tail_open = False
    _tail_tokens = 0
    _tail_free: torch.Tensor | None = None
    # Tail slots handed out and not yet freed.
    tail_outstanding = 0

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self.clear()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1,
            (self.size if self._tail_base is None else self._tail_base) + 1,
            dtype=torch.int64,
            device=self.device,
        )
        self.free_group = None
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
        if self._tail_base is not None:
            self._reset_tail_free()

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        if self._tail_open:
            return len(self.free_pages) + len(self.release_pages) + len(self._tail_free)
        return len(self.free_pages) + len(self.release_pages)

    # -- elastic tail --------------------------------------------------------

    def enable_elastic_tail(self, base_size: int) -> None:
        """Keep slots above ``base_size`` out of circulation until open_tail()."""
        if not 0 < base_size < self.size:
            raise ValueError(f"tail base {base_size} outside (0, {self.size})")
        if self.need_sort:
            raise ValueError("the elastic tail does not support sorted free lists")
        if self.available_size() != self.size:
            raise RuntimeError("the elastic tail must be enabled on an empty pool")
        self._tail_base = int(base_size)
        self._tail_open = False
        self._tail_tokens = 0
        self.clear()

    def _reset_tail_free(self) -> None:
        self.tail_outstanding = 0
        self._tail_free = torch.arange(
            self._tail_base + 1,
            self._tail_base + self._tail_tokens + 1,
            dtype=torch.int64,
            device=self.device,
        )

    def open_tail(self, num_tokens: int) -> None:
        """Hand out the first ``num_tokens`` tail slots from now on."""
        assert self._tail_base is not None and self._tail_tokens == 0
        assert 0 < num_tokens <= self.size - self._tail_base
        self._tail_tokens = int(num_tokens)
        self._reset_tail_free()
        self._tail_open = True

    def close_tail(self) -> None:
        """Stop handing out tail slots; those in use stay valid until freed."""
        self._tail_open = False

    def reopen_tail(self) -> None:
        """Undo close_tail() while the tail is still backed."""
        assert self._tail_tokens > 0
        self._tail_open = True

    def drop_tail(self) -> None:
        """Forget the tail once nothing holds a slot of it."""
        assert not self._tail_open and self.tail_outstanding == 0
        self._tail_tokens = 0
        self._reset_tail_free()

    def tail_slots_in_use(self) -> torch.Tensor:
        """The tail slots handed out and not yet freed, ascending."""
        in_use = torch.ones(self._tail_tokens, dtype=torch.bool, device=self.device)
        in_use[self._tail_free - self._tail_base - 1] = False
        return in_use.nonzero().flatten() + self._tail_base + 1

    def unavailable_size(self) -> int:
        """Slots that are neither free to hand out nor held: the tail that is
        not open, less what is still in use from it."""
        if self._tail_base is None:
            return 0
        tail = self.size - self._tail_base
        if self._tail_open:
            return tail - self._tail_tokens
        return tail - self.tail_outstanding

    def _alloc_with_tail(self, need_size: int):
        num_low = len(self.free_pages)
        num_tail = need_size - num_low
        if num_tail > len(self._tail_free):
            return None
        select_index = torch.cat((self.free_pages, self._tail_free[:num_tail]))
        self.free_pages = self.free_pages[num_low:]
        self._tail_free = self._tail_free[num_tail:]
        self.tail_outstanding += num_tail
        return select_index

    def _take_tail_frees(self, free_index: torch.Tensor) -> torch.Tensor:
        """Return freed tail slots to the tail and pass the rest on. Reads the
        ids, so it runs only while tail slots are in use."""
        is_tail = free_index > self._tail_base
        tail_index = free_index[is_tail]
        if tail_index.numel() == 0:
            return free_index
        self._tail_free = torch.cat((self._tail_free, tail_index))
        self.tail_outstanding -= tail_index.numel()
        assert self.tail_outstanding >= 0, "tail slot freed twice"
        return free_index[~is_tail]

    def get_all_free_pages(self):
        free_pages = super().get_all_free_pages()
        if self._tail_open:
            return torch.cat((free_pages, self._tail_free))
        return free_pages

    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            if self._tail_open:
                return self._alloc_with_tail(need_size)
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.free_group is None:
            if self.tail_outstanding:
                free_index = self._take_tail_frees(free_index)
            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(self._copy_for_free_group(free_index))

    def free_page_ids(self, page_ids: torch.Tensor):
        # page_size == 1: page ids are token ids.
        self.free(page_ids)

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        return self._kvcache.get_cpu_copy(
            indices,
            mamba_indices=mamba_indices,
            req_pool_index=req_pool_index,
        )

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu,
            indices,
            mamba_indices=mamba_indices,
            req_pool_index=req_pool_index,
        )
