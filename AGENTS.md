# OpenMettle fork instructions

Read [OPENMETTLE.md](OPENMETTLE.md) and
[OPENMETTLE-NOTICES.md](OPENMETTLE-NOTICES.md) before changing this fork.

- Work on feature branches based on `openmettle/rdna4` and target that branch in
  pull requests. `main` retains upstream history. Do not upgrade the upstream
  pin as an incidental part of another change.
- Keep architecture-independent changes portable. Introduce other maintained
  architecture branches only when needed; do not duplicate shared implementations.
- Engine changes and regression tests belong here. Build recipes, model profiles,
  benchmarks and qualification evidence belong in `OpenMettle/openmettle-serving`.
- The imported source already includes patches 0001–0033. Never apply the frozen
  serving patch series a second time or keep an independently edited patch copy.
- Preserve upstream and imported copyright, licence and provenance notices.
- Keep `SGLANG_GFX1201_*` feature switches off by default. Enable features only
  through an explicitly qualified serving profile.
- Run relevant CPU tests for changed logic; use Python `-I -B` where the test
  entry point supports it. GPU changes also need matched correctness, quality
  and performance qualification on the intended hardware before rollout.
- Report which checks actually ran. Source-tree equality is not GPU qualification.
- Preserve existing work and dated evidence. Owner review controls merges,
  releases and deployment. Do not force-push shared branches.

Nested AGENTS.md files and upstream contribution requirements continue to apply.
