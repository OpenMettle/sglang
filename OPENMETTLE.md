# OpenMettle SGLang

This fork maintains SGLang engine changes used by OpenMettle. The initial
`openmettle/rdna4` branch imports the 33 AMD RDNA4 patches from
[`OpenMettle/openmettle-serving`](https://github.com/OpenMettle/openmettle-serving)
at upstream commit `fcb3274646e43c16d70fd83806ebb3a5134d5e81`.

## Branches

- `main` retains upstream history for integration and comparison.
- `openmettle/rdna4` is the maintained AMD RDNA4 working branch, including elastic
  KV tail support. Create feature branches from it and target it explicitly in
  pull requests.
- Add an `openmettle/nvidia` or another architecture branch when there are actual
  architecture-specific changes to maintain. Keep shared features portable;
  prefer upstream contributions and focused shared commits over independent
  implementations on each branch.

The repository name stays `sglang` because the fork can support multiple
architectures and models. A branch identifies source development; a qualified
serving profile identifies a supported model, hardware and image combination.
No NVIDIA-specific branch or qualification is claimed by this import.

## Repository responsibilities

Engine implementation and engine regression tests belong here. Curated builds,
dependency pins, model profiles, benchmarks and qualification records belong in
`openmettle-serving`. Model profiles can select different weight revisions,
quantisations, backends and resource budgets without requiring separate serving
repositories. Kubernetes deployment packaging belongs in `openmettle-deploy`.

See the serving repository's
[migration record](https://github.com/OpenMettle/openmettle-serving/blob/main/docs/REPOSITORY-MIGRATION-20261116.md)
for per-patch provenance and exact source-tree verification. The frozen patch
files remain available for the previous workstation build. **Do not apply them
again on top of this branch.** Future builds should pin a full fork commit.

The migration preserves source behaviour. It does not establish a new image's
GPU performance, quality or deployment status. Keep RDNA4 feature switches off
by default and qualify each intended serving profile before rollout.

Upstream README, licence and history are retained. The imported work's additional
attribution is in [OPENMETTLE-NOTICES.md](OPENMETTLE-NOTICES.md).
