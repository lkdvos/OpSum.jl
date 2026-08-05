# research/

Background notes and algorithm studies that inform OpSum.jl's development. Not part of the package.

- [itensor-mpograph-construction.md](itensor-mpograph-construction.md) — how
  [ITensorMPOConstruction.jl](https://github.com/ITensor/ITensorMPOConstruction.jl) builds an exact
  minimal-bond-dimension MPO via a persistent bipartite `MPOGraph`, focusing on the site-to-site
  bookkeeping, with a worked example and a mapping onto OpSum's `_irrep_bipartite` frontier sweep.
- [port-handoff.md](port-handoff.md) — the brief for porting that architecture onto the non-abelian ITO
  machinery: required reading, the additive-flux → fusion-outcome mapping, milestones, watch-outs.
- [persistent-graph-mpo.md](persistent-graph-mpo.md) — the resulting design: `ITOGraph`, the five-phase
  `_at_site!` sweep, interned suffix classes and lazy insertion (which together make it linear in `N`
  for finite-range models), the pending↔started class collision, and the cost measurements.
- [infinite-mpo.md](infinite-mpo.md) — extending the construction to infinite chains with a repeating
  unit cell: what survives unchanged, why translation covariance forces a canonical ordering (the
  `L = 1` hazard), how the unit cell is closed, and a design for exponentially decaying terms.
- [exp-decay-handoff.md](exp-decay-handoff.md) — the brief for building that last part: a geometric
  channel as a suffix class with a self-loop, and the partition-refinement class naming a cyclic tail
  needs.
