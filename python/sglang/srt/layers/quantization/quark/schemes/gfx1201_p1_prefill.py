"""P1 prefill route for gfx1201: MXFP4 prefill rows on a lossless FP8 expansion.

Ported from the serving repository's P1 route (P1-NUMERICAL-CONTRACT-v1,
setting fp8-g128-b101-m1024-v1, operation 93). Admitted when
1024 <= M <= 2048 on the five declared N/K layouts and never under graph
capture. Decode and verify rows, and every other shape, stay on the MXFP4
kernels. The kernel bodies (gfx1201_p1_fp8_kernels, byte-identical to the
accepted p1_fp8_kernels.py), launch configurations, grids and argument order
are unchanged; only the row count varies, and each output row's arithmetic is
independent of it.

Every admitted shard is expanded once at load and its guard flags must all be
zero. The packed weights stay resident and are expanded again into one
reusable per-card scratch on each admitted call. All state lives in this
module; nothing is attached to a layer. Importing this module is CPU-only.

Changes from the source route: the kernel import path, route markers go to the
logger, and validate_layer(..., rebind=True) re-validates a layer whose packed
weights were moved without changing their bytes (patch 0007 packs the GDN
input projections into one tensor after loading); and a layer stored as kq3
records (gfx1201_mxfp4_kq3) can be expanded from the records
(validate_records, apply(..., records=...)) by gfx1201_p1_kq3, whose output is
compared at load with the expansion above.
"""

import logging

logger = logging.getLogger(__name__)

SETTING = "fp8-g128-b101-m1024-v1"
MIN_ROWS, MAX_ROWS = 1024, 2048
SHAPES = ((17408, 5120), (8192, 5120), (5120, 3072), (7168, 5120), (5120, 8704))
QUANTIZER_CONFIG = {"BLOCK": 128, "num_warps": 1, "num_stages": 1}
GEMM_CONFIG = {
    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32, "needs_masking": False, "num_warps": 4, "num_stages": 1,
}
EXPANSION_CONFIG = {"num_warps": 8, "num_stages": 1}
MAX_WEIGHT = max(n * k for n, k in SHAPES)
MAX_BLOCKS = max((n // 128) * (k // 128) for n, k in SHAPES)
MAX_K = max(k for _, k in SHAPES)
WARM_ROWS = (MIN_ROWS, MIN_ROWS + 1)  # Triton specializes integer M on divisibility by 16.
ACTIVE = None  # the one route of this serving process


def scratch_bytes():
    """Persistent per-card scratch; allocator overhead is measured separately."""
    return MAX_WEIGHT + 8 * MAX_BLOCKS + MAX_ROWS * MAX_K + 4 * MAX_ROWS * (MAX_K // 128)


class PrefillRoute:
    def __init__(self, torch, setting, emit=None):
        if setting != SETTING:
            raise ValueError("unknown P1 prefill setting")
        self.torch, self.emit = torch, emit
        self.scratch, self.validated, self.warmed = {}, {}, set()
        self.calls, self.rows, self.seen = {}, {}, set()
        self.records, self.from_records = {}, 0

    def admits_shape(self, n, k):
        return (n, k) in SHAPES

    def admits(self, m, n, k):
        return type(m) is int and MIN_ROWS <= m <= MAX_ROWS and (n, k) in SHAPES

    def _runtime(self, device):
        if type(device) is not int or device not in (0, 1):
            raise ValueError("P1 requires card 0 or 1")
        import triton

        from sglang.srt.layers.quantization.quark.schemes import (
            gfx1201_p1_fp8_kernels as kernels,
        )

        if self.torch.cuda.current_device() != device:
            raise ValueError("P1 wrong current card")
        target = triton.runtime.driver.active.get_current_target()
        if target.backend != "hip" or target.arch != "gfx1201":
            raise ValueError("P1 admits only gfx1201 HIP")
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("P1 prefill route never runs under graph capture")
        return triton, kernels

    def _buffers(self, device):
        if device not in self.scratch:
            torch = self.torch
            # One buffer per logical tensor: the launches require distinct storage.
            self.scratch[device] = dict(
                weight=torch.empty((MAX_WEIGHT,), dtype=torch.float8_e4m3fn, device=device),
                block_scales=torch.empty((MAX_BLOCKS,), dtype=torch.float32, device=device),
                flags=torch.empty((MAX_BLOCKS,), dtype=torch.int32, device=device),
                x_q=torch.empty((MAX_ROWS * MAX_K,), dtype=torch.float8_e4m3fn, device=device),
                x_scales=torch.empty((MAX_ROWS * (MAX_K // 128),), dtype=torch.float32, device=device))
        return self.scratch[device]

    def _views(self, device, m, n, k):
        b = self._buffers(device)
        blocks = (n // 128) * (k // 128)
        return (b["weight"][:n * k].view(n, k), b["block_scales"][:blocks].view(n // 128, k // 128),
                b["flags"][:blocks].view(n // 128, k // 128), b["x_q"][:m * k].view(m, k),
                b["x_scales"][:m * (k // 128)].view(m, k // 128))

    def _check(self, value, shape, dtype, device):
        if (tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous()
                or value.device.type != "cuda" or value.device.index != device):
            raise ValueError("P1 tensor shape/dtype/contiguity/card mismatch")

    def _check_layer(self, layer, n, k, device):
        self._check(layer.weight, (n, k // 2), self.torch.uint8, device)
        self._check(layer.weight_scale, (n, k // 32), self.torch.uint8, device)

    def _identity(self, layer):
        w, s = layer.weight, layer.weight_scale
        return (w.data_ptr(), s.data_ptr(), w._version, s._version, tuple(w.shape), tuple(s.shape))

    def _expand(self, kernels, triton, layer, n, k, weight, block_scales, flags):
        return kernels._expand_mxfp4_to_block_fp8[(triton.cdiv(n, 128), k // 128)](
            layer.weight, layer.weight_scale, weight, block_scales, flags,
            N=n, K=k, **EXPANSION_CONFIG,
        )

    def _records_identity(self, records):
        rec = records.rec
        return (rec.data_ptr(), rec._version, tuple(rec.shape), records.n, records.k)

    def _expand_records(self, records, n, k, weight, block_scales, flags):
        from sglang.srt.layers.quantization.quark.schemes import gfx1201_p1_kq3

        if (records.n, records.k) != (n, k) or not records.rec.is_contiguous():
            raise ValueError("P1 records shape/contiguity mismatch")
        return gfx1201_p1_kq3.expand(records.rec, n, k, weight, block_scales, flags, **EXPANSION_CONFIG)

    def _quantize(self, kernels, x, x_q, x_scales, m, k):
        return kernels._per_token_group_quant_8bit[(m * (k // 128),)](
            x, x_q, x_scales, 128, 128, 1e-10, -448.0, 448.0, **QUANTIZER_CONFIG,
        )

    def _gemm(self, kernels, triton, x_q, weight, x_scales, block_scales, y, m, n, k):
        return kernels._w8a8_block_fp8_matmul[(triton.cdiv(m, 64) * triton.cdiv(n, 128),)](
            x_q, weight, y, x_scales, block_scales, m, n, k, 128, 128,
            x_q.stride(0), x_q.stride(1), weight.stride(1), weight.stride(0),
            y.stride(0), y.stride(1), x_scales.stride(0), x_scales.stride(1),
            block_scales.stride(1), block_scales.stride(0), **GEMM_CONFIG,
        )

    def is_validated(self, layer):
        return id(layer) in self.validated

    def validate_layer(self, layer, n, k, rebind=False):
        """Load time: expand once, require every guard flag zero, then warm the row-count variants.

        rebind=True re-validates a layer validated before whose packed weights moved.
        """
        torch = self.torch
        if (n, k) not in SHAPES:
            raise ValueError("P1 validation of an undeclared shape")
        device = layer.weight.device.index
        triton, kernels = self._runtime(device)
        if (id(layer) in self.validated) != rebind:
            raise RuntimeError("P1 layer validated twice" if not rebind else "P1 rebind of an unvalidated layer")
        self._check_layer(layer, n, k, device)
        weight, block_scales, flags, _, _ = self._views(device, MIN_ROWS, n, k)
        with torch.no_grad():
            self._expand(kernels, triton, layer, n, k, weight, block_scales, flags)
            bad = int(torch.count_nonzero(flags).item())  # synchronizes; load time only
        if bad:
            raise ValueError("P1 guard flags rejected %d blocks of an admitted shard" % bad)
        self.validated[id(layer)] = self._identity(layer)
        if device not in self.warmed:
            # Compile both M specializations of the quantizer and GEMM before traffic, on this
            # freshly expanded weight; outputs are discarded and nothing persists.
            for m in WARM_ROWS:
                x = torch.zeros((m, k), dtype=torch.bfloat16, device=device)
                y = torch.empty((m, n), dtype=torch.bfloat16, device=device)
                _, _, _, x_q, x_scales = self._views(device, m, n, k)
                self._quantize(kernels, x, x_q, x_scales, m, k)
                self._gemm(kernels, triton, x_q, weight, x_scales, block_scales, y, m, n, k)
                del x, y
            torch.cuda.synchronize(device)
            self.warmed.add(device)
            if self.emit is not None:
                self.emit("P1_PREFILL_READY", {"device": device, "scratch_bytes": scratch_bytes(),
                                               "warm_rows": list(WARM_ROWS), "setting": SETTING})

    def has_records(self, records):
        return id(records) in self.records

    def validate_records(self, records, n, k):
        """Load time, after validate_layer(records.canonical(), n, k): the expansion read from
        the records must be, bit for bit, the validated expansion of that canonical pair.

        The comparison buffers start as the complement of the reference, so a byte the
        records kernel does not write cannot compare equal.
        """
        torch = self.torch
        device = records.rec.device.index
        triton, kernels = self._runtime(device)
        canon = records.canonical()
        if self.validated.get(id(canon)) != self._identity(canon):
            raise RuntimeError("P1 records validated before their canonical pair")
        if id(records) in self.records:
            raise RuntimeError("P1 records validated twice")
        self._check(records.rec, tuple(records.rec.shape), torch.uint8, device)
        weight, block_scales, flags, _, _ = self._views(device, MIN_ROWS, n, k)
        with torch.no_grad():
            self._expand(kernels, triton, canon, n, k, weight, block_scales, flags)
            got_w = torch.empty_like(weight)
            got_s = torch.empty_like(block_scales)
            got_f = torch.full_like(flags, -1)
            got_w.view(torch.uint8).copy_(~weight.view(torch.uint8))
            got_s.view(torch.int32).copy_(~block_scales.view(torch.int32))
            self._expand_records(records, n, k, got_w, got_s, got_f)
            same = (int(torch.count_nonzero(flags).item()) == 0  # synchronizes; load time only
                    and torch.equal(got_f, flags)
                    and torch.equal(got_s.view(torch.int32), block_scales.view(torch.int32))
                    and torch.equal(got_w.view(torch.uint8), weight.view(torch.uint8)))
            del got_w, got_s, got_f
        # The comparison buffers go back to the device.
        torch.cuda.empty_cache()
        if not same:
            raise ValueError("P1 expansion from kq3 records differs from the validated expansion")
        self.records[id(records)] = self._records_identity(records)
        if ("kq3", n, k) not in self.seen and self.emit is not None:
            self.seen.add(("kq3", n, k))
            self.emit("P1_PREFILL_FROM_KQ3", {"device": device, "n": n, "k": k})

    def apply(self, layer, x, n, k, records=None):
        """One admitted prefill call: expand, quantize, GEMM; BF16 output in a fresh tensor.

        records: the layer's kq3 Records, validated by validate_records; the weight is then
        expanded from them and the canonical pair (layer) is not read."""
        torch = self.torch
        m = x.shape[0]
        if not self.admits(m, n, k):
            raise ValueError("P1 call outside its admission")
        device = x.device.index
        triton, kernels = self._runtime(device)
        if self.validated.get(id(layer)) != self._identity(layer):
            raise RuntimeError("P1 layer not validated at load or its packed weights changed")
        if records is not None and (
                layer is not records.canon or self.records.get(id(records)) != self._records_identity(records)):
            raise RuntimeError("P1 records not validated at load or changed")
        self._check_layer(layer, n, k, device)
        self._check(x, (m, k), torch.bfloat16, device)
        weight, block_scales, flags, x_q, x_scales = self._views(device, m, n, k)
        y = torch.empty((m, n), dtype=torch.bfloat16, device=device)
        if records is None:
            self._expand(kernels, triton, layer, n, k, weight, block_scales, flags)
        else:
            self._expand_records(records, n, k, weight, block_scales, flags)
        self._quantize(kernels, x, x_q, x_scales, m, k)
        self._gemm(kernels, triton, x_q, weight, x_scales, block_scales, y, m, n, k)
        if records is not None:
            self.from_records += 1
        self.calls[(n, k)] = self.calls.get((n, k), 0) + 1
        self.rows[m] = self.rows.get(m, 0) + 1
        if (n, k) not in self.seen and self.emit is not None:
            self.seen.add((n, k))
            self.emit("P1_PREFILL_ROUTE", {"device": device, "n": n, "k": k, "rows": m})
        return y

    def snapshot(self):
        snap = {"setting": SETTING, "validated_layers": len(self.validated), "warmed_devices": sorted(self.warmed),
                "calls": {"%dx%d" % key: value for key, value in sorted(self.calls.items())},
                "rows": {str(key): value for key, value in sorted(self.rows.items())}}
        if self.records:
            snap["kq3_records"] = {"validated_layers": len(self.records), "calls": self.from_records}
        return snap


def install(torch, setting):
    """Create and register the single route for this serving process."""
    global ACTIVE
    if ACTIVE is not None:
        raise RuntimeError("P1 prefill route is single-use per serving process")
    ACTIVE = PrefillRoute(torch, setting, emit=log_emit)
    return ACTIVE


def log_emit(label, value):
    import json

    logger.info("%s %s", label, json.dumps(value, sort_keys=True))
