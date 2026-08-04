# research/

Background notes and algorithm studies that inform OpSum.jl's development. Not part of the package.

- [persistent-graph-mpo.md](persistent-graph-mpo.md) — **the as-built design note for the default MPO
  sweep.** How the persistent bipartite graph walks the chain site by site, why interned suffix
  classes (§2.1) and lazy right-vertex insertion (§2.2) make it linear in `N` for finite-range
  models, and what the measured scaling is (§2.3). **Read §2.2 before changing anything in
  `irrepgraph.jl`**: the pending-versus-started suffix-class collision documented there is the one
  invariant a change is most likely to break silently.
- [itensor-mpograph-construction.md](itensor-mpograph-construction.md) — how
  [ITensorMPOConstruction.jl](https://github.com/ITensor/ITensorMPOConstruction.jl) builds an exact
  minimal-bond-dimension MPO via a persistent bipartite `MPOGraph`, focusing on the site-to-site
  bookkeeping, with a worked example. The primary spec the port was written against.
- [port-handoff.md](port-handoff.md) — the original handoff prompt for that port. Historical: useful
  for *why* things are scoped the way they are (what was deliberately left out), not as a
  description of the code as it now stands.
- [onsite-products.md](onsite-products.md) — why the symbolic `Prod`/`Pow` scaffolding was removed,
  what still works today (build the product as a `TensorMap` and `project` it), and what adding
  symbolic on-site products back would actually require.
