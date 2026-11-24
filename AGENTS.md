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

## Ticket and sprint coordination

[Project 1](https://github.com/orgs/OpenMettle/projects/1) and the [delivery tracker](https://github.com/OpenMettle/openmettle-core/issues/1) are the live work index. This policy applies to every ticket, including delegated work.

- **Accountability:** the main (coordinating) agent owns task selection, scope control and tracker reconciliation. Delegating work does not transfer these duties.
- **Check before work:** before implementing or delegating, after resuming or recovering context, and before starting another ticket, fetch the live issue, its Project 1 fields and the relevant sprint record. Check the acceptance criteria, dependencies, recent relevant comments and existing PRs. Then briefly state the selected issue, its sprint or goal and the bounded scope.
- **Scope:** give each subagent an explicit assignment and the evidence it must return. Record unrelated discoveries and out-of-scope review findings as follow-up issues in Project 1, in an appropriate sprint or the backlog. Preserve agreed priorities, historical sprint assignments and capacity decisions. Follow explicit user-directed scope changes and record them on the sprint record.
- **Tracking updates:** when work starts, reaches review or is handed off, reconcile the owning issue, its Project 1 status and the relevant sprint record. Record PR links, review findings, the verification actually run, remaining gaps and next steps. Read each update back to confirm that it persisted.
- **Completion:** close an issue or set it to Done only when its full acceptance criteria and any applicable owner gates are met. Keep partial increments and unfinished parent issues open.
- **Access failures:** if a tracker read or write fails, report it and list the outstanding updates. Continue only clearly authorized work. Do not guess missing state or claim that synchronization succeeded.

Nested AGENTS.md files and upstream contribution requirements continue to apply.
