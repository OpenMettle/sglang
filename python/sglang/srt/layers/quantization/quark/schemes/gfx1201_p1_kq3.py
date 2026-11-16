# SPDX-License-Identifier: Apache-2.0
"""P1 weight expansion read from kq3 records (SGLANG_GFX1201_P1_FROM_KQ3).

gfx1201_p1_fp8_kernels._expand_mxfp4_to_block_fp8 reads the canonical MXFP4
pair, which a kq3 layer no longer keeps: gfx1201_mxfp4_kq3 rebuilds it into a
scratch before every P1 call. The kernel here writes the same expansion, block
scales and guard flags from the records themselves. Its arithmetic is that
kernel's, statement for statement; only the addressing differs. One program is
still one N128 x K128 block: two N64 record tiles by four K32 records.

Importing this module needs Triton; the P1 route imports it on first use.
"""

import triton
import triton.language as tl


@triton.jit
def _expand_kq3_to_block_fp8(Records, Expanded, BlockScales, GuardFlags, K: tl.constexpr):
    """Records (tiles of 64 rows, K / 32, 1088) of N rows, N a multiple of 128 ->
    what _expand_mxfp4_to_block_fp8 writes for the canonical pair they unpack to."""
    bn = tl.program_id(0)
    bk = tl.program_id(1)
    r = tl.arange(0, 128)
    j = tl.arange(0, 16)  # canonical dword of the block: K 8j .. 8j + 7
    row = r % 64
    rec = (2 * bn + r // 64)[:, None] * (K // 32) + (bk * 4 + j // 4)[None, :]
    # A record stores scale - 1 at byte 4 * (row % 16) + row // 16; the canonical byte wraps.
    scales = (tl.load(Records + rec * 1088 + 1024 + (4 * (row % 16) + row // 16)[:, None]).to(tl.int32) + 1) & 255
    low = tl.min(tl.min(scales, 1), 0)
    high = tl.max(tl.max(scales, 1), 0)
    invalid_zero = tl.max(tl.max((scales == 0).to(tl.int32), 1), 0)
    invalid_reserved = tl.max(tl.max((scales == 255).to(tl.int32), 1), 0)
    flags = invalid_zero | (invalid_reserved << 1) | ((high - low > 14).to(tl.int32) << 2)
    tl.store(GuardFlags + bn * (K // 128) + bk, flags)
    valid = flags == 0
    exponent = tl.minimum(high - 127, low - 127 + 8)
    safe_exponent = tl.where(valid, exponent, 0)
    block_scale = ((safe_exponent + 127) << 23).to(tl.float32, bitcast=True)
    tl.store(BlockScales + bn * (K // 128) + bk, block_scale, mask=valid)

    # Canonical dword c of a row and K32 step is record lane row % 16 + 16 * (c % 2), lane
    # dword (row // 16) * 2 + c // 2; its nibbles i and i + 4 are codes 2i and 2i + 1.
    c = j % 4
    rdw = ((row % 16)[:, None] + 16 * (c % 2)[None, :]) * 8 + (row // 16)[:, None] * 2 + (c // 2)[None, :]
    v = tl.load(Records.to(tl.pointer_type(tl.uint32)) + rec * 272 + rdw)
    kk = tl.arange(0, 8)
    shift = (4 * (kk // 2 + 4 * (kk % 2))).to(tl.uint32)
    code = ((v[:, :, None] >> shift[None, None, :]) & 15).to(tl.int32)
    m = code & 7
    delta = tl.where(valid, scales - 127 - safe_exponent, 0)[:, :, None]
    normal_code = tl.where(m == 1, 48, m * 4 + 48) + delta * 8
    twice_magnitude = tl.where(m < 4, m, tl.where(m < 6, (m - 2) * 2, (m - 4) * 4))
    subnormal_code = twice_magnitude << (delta + 8)
    magnitude_code = tl.where(m == 0, 0, tl.where(normal_code >= 8, normal_code, subnormal_code))
    bits = (magnitude_code | ((code & 8) << 4)).to(tl.uint8)
    tl.store(
        Expanded + ((bn * 128 + r) * K)[:, None, None] + (bk * 128 + j * 8)[None, :, None] + kk[None, None, :],
        bits.to(Expanded.dtype.element_ty, bitcast=True),
        mask=valid,
    )


def expand(rec, n, k, weight, block_scales, flags, **config):
    """Launch on the grid of _expand_mxfp4_to_block_fp8: one program per N128 x K128 block."""
    if n % 128 or k % 128 or rec.dim() != 3 or tuple(rec.shape[1:]) != (k // 32, 1088) or rec.shape[0] * 64 < n:
        raise ValueError("P1 kq3 expansion of an unsupported record layout")
    return _expand_kq3_to_block_fp8[(n // 128, k // 128)](rec, weight, block_scales, flags, K=k, **config)
