# Imported RDNA4 source notices

The engine import applies patches 0001–0033 from `OpenMettle/openmettle-serving`
revision `8a53e9c2bf2837d4cd9296b18627067be98f27e8` without changing their source
bytes. Each import commit identifies the patch, SHA-256 and source history.
The upstream root Apache-2.0 licence and all existing file notices are retained.

Original serving-project work is covered by the retained
[MIT notice](LICENSES/OpenMettle-serving-MIT.txt). This does not replace the terms
of copied SGLang, vLLM, AMD or other third-party code.

## AMD AITER

The imported `gfx1201_mxfp4_kernels.py` derives from `ROCm/aiter` revision
`32214c295ea2eec1055bdb3467d2eee7289257be` and retains its MIT SPDX marker and
AMD copyright notice. Patches 0010, 0019 and 0020 reuse its quantizer in other
kernels. The [AMD MIT notice](LICENSES/AITER-MIT.txt) is retained for those
derived portions, regardless of the surrounding file's SPDX marker.

## Flash Linear Attention

Patch 0010's `gfx1201_gdn_norm_quant.py` identifies its kernel body as derived
from SGLang's `python/sglang/kernels/ops/attention/fla/layernorm_gated.py`.
At the pinned upstream revision that source includes this attribution:

> Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
>
> Copyright (c) 2024, Tri Dao.
>
> Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html

That attribution is retained here for the derived kernel without modifying the
imported engine bytes. The [FLA MIT licence](LICENSES/FLA-MIT.txt) was retrieved
from `fla-org/flash-linear-attention` on 16 November 2026; its Git blob is
`7ced50191b7f5f0eec4e14114d2bc613922bc3b2`. This supplements the earlier serving
notice's unresolved attribution check; it is not a complete dependency audit.

## Upstream contributions

Patches 0028 and 0029 port SGLang pull requests #47057 and #47058. Their authorship
remains attributed to the upstream contributors and the original patch history;
the import commits record the migration, not new authorship of those changes.

The source repository's `THIRD-PARTY.md` at the recorded revision retains the
broader research and historical-fixture provenance. BetterBench, model weights
and the historical serving overlays were not copied into this fork.
