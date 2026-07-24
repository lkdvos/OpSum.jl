# research/

Background notes and algorithm studies that inform OpSum.jl's development. Not part of the package.

- [itensor-mpograph-construction.md](itensor-mpograph-construction.md) — how
  [ITensorMPOConstruction.jl](https://github.com/ITensor/ITensorMPOConstruction.jl) builds an exact
  minimal-bond-dimension MPO via a persistent bipartite `MPOGraph`, focusing on the site-to-site
  bookkeeping, with a worked example and a mapping onto OpSum's `_irrep_bipartite` frontier sweep.
