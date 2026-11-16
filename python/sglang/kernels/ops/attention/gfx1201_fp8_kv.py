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

The cast itself: Triton lowers ``fp8e4nv -> bf16`` on gfx1201 to integer arithmetic (about
nine vector instructions per value) although the ISA has ``v_cvt_f32_fp8``. For an
``float8_e4m3fn`` cache the kernels decode with that instruction instead (``DECODE``): every
e4m3fn value has at most four significant bits, so the f32 result's high half is the exact
bf16, and the outputs are bitwise equal to the Triton cast's. ``DECODE_HW`` converts one
value per instruction on any tile (extend, grouped decode); ``DECODE_PAIRS`` reads 16-bit
pairs and packs two bf16 per ``v_perm_b32`` (target verify). On an R9700 at Qwen3.8 TP2
shapes (16 layers, DRAM-fed) this takes fp8 target verify from 1.9-2.5x the bf16 time to
0.96-1.04x (8k-123k), grouped decode from 1.8x to 0.80x (47k) and extend from 2.4-2.5x to
1.15-1.18x (8k-88k prefix, 2048-token chunk). The asm is
declared impure on purpose: a pure (or Triton-cast) decode lets the layout pass move the
dot-operand conversion above it to shuffle the narrower fp8 data, and the decode then runs
on the replicated operand (6x the conversions at the B1 verify tile). Impure, it runs once
per loaded value, before the shuffle.
"""

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from sglang.srt.utils import get_bool_env_var, is_hip

FP8_TYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2)
DECODE_TO, DECODE_HW, DECODE_PAIRS = 0, 1, 2
_DECODE_NAMES = {"to": DECODE_TO, "hw": DECODE_HW, "pairs": DECODE_PAIRS}


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


def decode(q: torch.Tensor, k_buffer: torch.Tensor, v_buffer: torch.Tensor, pairs: bool = False) -> int:
    """The ``DECODE`` constexpr for a kernel reading these caches with a bf16 ``q``: a
    hardware decode for e4m3fn caches under the switch (pairs where the kernel supports
    them), else Triton's cast. SGLANG_GFX1201_FP8_KV_DECODE=to|hw|pairs overrides
    (measurement only)."""
    if not (
        applies(k_buffer, v_buffer)
        and k_buffer.dtype == torch.float8_e4m3fn
        and q.dtype == torch.bfloat16
    ):
        return DECODE_TO
    mode = _DECODE_NAMES[os.environ.get("SGLANG_GFX1201_FP8_KV_DECODE", "pairs")]
    return DECODE_HW if mode == DECODE_PAIRS and not pairs else mode


@triton.jit
def e4m3fn_to_bf16(x):
    """float8_e4m3fn tensor (any layout) -> bf16, exact: v_cvt_f32_fp8, keep the high half."""
    f = tl.inline_asm_elementwise(
        asm="v_cvt_f32_fp8_e64 $0, $1",
        constraints="=v,v",
        args=[x.to(tl.uint8, bitcast=True).to(tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    return (f >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def e4m3fn_pairs_to_bf16(h):
    """[R, C] uint16, each two float8_e4m3fn bytes (low byte first) -> [R, 2C] bf16, exact.
    Two u16 per register: v_cvt_f32_fp8 each byte, v_perm_b32 packs the f32 high halves."""
    lo, hi, _ = tl.inline_asm_elementwise(
        asm="v_cvt_f32_fp8_e64 $0, $3\n"
        "v_cvt_f32_fp8_e64 $1, $3 byte_sel:2\n"
        "v_perm_b32 $0, $1, $0, 0x07060302\n"
        "v_cvt_f32_fp8_e64 $1, $3 byte_sel:1\n"
        "v_cvt_f32_fp8_e64 $2, $3 byte_sel:3\n"
        "v_perm_b32 $1, $2, $1, 0x07060302",
        constraints="=&v,=&v,=&v,v",
        args=[h],
        dtype=(tl.bfloat16, tl.bfloat16, tl.bfloat16),
        is_pure=False,
        pack=2,
    )
    return tl.reshape(tl.join(lo, hi), [h.shape[0], h.shape[1] * 2])
