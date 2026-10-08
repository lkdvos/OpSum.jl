# research/

Background notes and algorithm studies that inform OpSum.jl's development.
Not part of the package.

- [overview-pipeline.md](overview-pipeline.md) — the six stages from on-site operators to emitted tensors, what each stage needs from the physical problem, and where the design is under pressure.
- [overview-types.md](overview-types.md) — every type, grouped as on-site operators, latticeless terms, lattice-bound sums, outputs and internals, with their fields and constructors.
- [overview-interface.md](overview-interface.md) — what a user writes today, step by step, with what works well and where it is thin.
- [persistent-graph-mpo.md](persistent-graph-mpo.md) — the finite-chain sweep design: `ITOGraph`, the five-phase `_at_site!` sweep, interned suffix classes and lazy insertion (which together make it linear in `N` for finite-range models), and the pending↔started class collision.
- [infinite-mpo.md](infinite-mpo.md) — extending the construction to infinite chains with a repeating unit cell: what survives unchanged, why translation covariance forces a canonical ordering, how the unit cell is closed, and — in §7 — exponentially decaying terms.
  §8 collects open follow-ups.
- [local-operators.md](local-operators.md) — unplaced multi-site `LocalOperator`s with `couple`/`dot`, and a lattice-carrying `OperatorSum` as the one place spaces enter; implemented.
- [operator-bases.md](operator-bases.md) — why the implicit matrix-unit alphabet inflates bond dimensions (#38), and the agreed fix: an explicit `Basis` used to project operators in and instantiate the MPO out.
  Includes the N₂ evidence and, as an appendix, the cover tie-break and re-blocking findings; not implemented.
