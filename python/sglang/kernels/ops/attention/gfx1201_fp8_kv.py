# SPDX-License-Identifier: Apache-2.0
"""SGLANG_GFX1201_FP8_KV_DEQUANT=1 on gfx1201: read an fp8 KV cache as bf16.

With ``--kv-cache-dtype fp8_e4m3`` the stock Triton attention kernels cast ``q`` and the
softmax weights ``p`` down to the cache's fp8 type and run fp8 x fp8 dots, so query
precision and every weight below e4m3's smallest subnormal (2^-9 of the row maximum) are
lost. The target-verify kernel (``verify_attention_gfx1151``) refuses fp8 caches outright,
which sends DFlash verification to ``extend_attention_fwd`` (30-40x slower per layer at
long prefixes on gfx1201).

With this switch the cache stays fp8 (half the bytes of bf16), but the extend, grouped
decode and target-verify kernels cast each loaded K/V tile up to bf16 and keep ``q`` and
``p`` in bf16, the same operands as a bf16 cache. Per-layer ``k_scale``/``v_scale`` apply
as before. Only the storage rounding of K and V remains.
"""

from functools import lru_cache

import torch

from sglang.srt.utils import get_bool_env_var, is_hip

FP8_TYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2)


@lru_cache(maxsize=1)
def enabled() -> bool:
    """SGLANG_GFX1201_FP8_KV_DEQUANT=1 and the device is gfx1201."""
    if not (is_hip() and get_bool_env_var("SGLANG_GFX1201_FP8_KV_DEQUANT")):
        return False
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1201"
    except Exception:
        return False


def applies(k_buffer: torch.Tensor, v_buffer: torch.Tensor) -> bool:
    """The switch is on and both caches hold the same fp8 type."""
    return enabled() and k_buffer.dtype in FP8_TYPES and v_buffer.dtype == k_buffer.dtype
