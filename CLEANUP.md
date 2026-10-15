# Guided review of OpSum.jl — decision log

**Delete before merging.** Its substance goes into the PR description.

Base: `722754e` (main). Branch/worktree: `cleanup`.

## Process

Per chunk: present (purpose, reading order, design decisions, proposals, questions) → user decides → decisions
recorded here → implemented by a subagent as verified commits → report reviewed → next chunk. Moves are separate
commits from behaviour changes. Verification: Runic clean, full `Pkg.test()`, old-vs-new equivalence script against
a worktree of the base commit (grown chunk by chunk), allocation / time-to-first-call checks where relevant.
Performance claims need same-node A/B runs submitted by the user.

## Global map

Layers (include order in `src/OpSum.jl`; arrows = "is used by"):

```
utility/ (linalg, memo)   datastructures/ (bipartite, connectedcomponents)   compression/algorithms.jl (strategy types)
        │                                                                          │
lattices.jl ──► algebra/  letters: IrrepTensorOperators(IrrepOperator) → irrepkey(ITOKey) → siteoperator(SiteOperator)
                          terms:   irrepalgebra(Term, PendingTerm, Terms, opsum, couple) → irrepinstantiate
                          blocks:  localoperator(LocalOperator) → irrepprojection(_project_units) → basis(Basis, project)
                          misc:    builders(adjoint!) → products(*, _recouple, _resolve) → letterproducts → stockbases
        ──► compression/irreptermtable (canonicalize! + ITOTermTable) ─► infinite/expterms (ExpSum, channels)
        ──► operatorsum (OperatorSum) ─► compression/symbolicmpo (mpo, SymbolicMPO)
        ──► compression/irrepinterning → irrepgraph (ITOGraph, sweep) → irrepgraph_vc | irrepgraph_svd → metric
        ──► infinite/infinitechain → infinitegraph (_infinite_window)
        ──► compression/irrepmpo (_compress, mpo_terms, tensors) → jordanmpo
        ──► timeevolution/kronpower → taylormpo (TaylorEntry, TaylorMPO) → taylor (the passes)
ext/OpSumMPSKitExt (instantiate → MPSKit MPOs, TaylorCluster)
```

Main path: `opsum(lat, ops...)` → `OperatorSum` (`_checkinsert`) → `mpo(H, alg)` (symbolicmpo.jl) → `_compress`
(irrepmpo.jl) → `ITOTermTable` → `_irrep_channels` → `ITOGraph` + `_at_site!` per site → `_bond_basis!`
(VC / SVD) → `SymbolicMPO(lat, Ws, bondsectors, starts, finishes)` → `irrep_mpo_tensors` / `jordan_mpo_tensors` /
`instantiate` (ext).

Core types: `IrrepOperator` (letter) · `ITOKey` (letter + outgoing bond charge) · `SiteOperator` (sparse sum of
letters) · `Term`/`Terms`/`PendingTerm` · `LocalOperator` (unplaced K-site) · `Basis` · `OperatorSum` · `ExpSum`/
`ExpChannel` · `ITOTermTable` · `ITOGraph` · `SymbolicMPO` · `JordanLayout` · `TaylorEntry`/`TaylorMPO`/`TaylorWork`.

Global mutable state: 5 `Dict{Any,Any}` / `IdDict` memo caches (stock bases, adjoint, recouple, pending products,
letter products) under one `ReentrantLock` (`utility/memo.jl`); never cleared.

Sizes (lines): src 5.8k (algebra 2.0k, compression 1.8k, timeevolution 0.8k, infinite 0.7k), test 5.6k, examples
2.0k, benchmark 0.6k, scripts 0.4k, research 0.2k. Largest: irrepalgebra.jl 586, irrepgraph.jl 514, expterms.jl 424.

## Chunks

| # | Chunk | Files | Status |
|---|-------|-------|--------|
| 0 | Top-down tour of the main path | — | pending |
| 1 | Foundations | utility/, datastructures/, compression/algorithms.jl, lattices.jl | pending |
| 2 | Letters | irreptensoroperators, irrepkey, siteoperator, irrepinstantiate | pending |
| 3 | Terms | irrepalgebra.jl, canonicalize! (irreptermtable.jl) | pending |
| 4 | Bases and projection | basis, irrepprojection, stockbases, builders (adjoint) | pending |
| 5 | Unplaced operators and products | localoperator, products, letterproducts | pending |
| 6 | Container and channels | operatorsum, infinite/expterms | pending |
| 7 | The sweep | irreptermtable (table), irrepinterning, irrepgraph | pending |
| 8 | Bond bases | irrepgraph_vc, irrepgraph_svd, metric | pending |
| 9 | Output | symbolicmpo, irrepmpo, jordanmpo, ext/ | pending |
| 10 | Infinite chains | infinite/infinitechain, infinitegraph | pending |
| 11 | Time evolution | timeevolution/ | pending |
| 12 | Organisation revisit | — | pending |
| 13 | Tests | test/ | pending |
| 14 | Examples, benchmarks, scripts, docs, research | | pending |

## Decisions

**D1 (convention) Underscore names are rare and never reach tests.** Internal helpers get plain names; `_` only
for genuinely private pieces (inner methods, scratch helpers). Tests use exported or plain-named internals only.
Applied per chunk: each chunk renames its `_` functions that tests reach and rewrites those tests.

**D2 (convention) No compatibility constraints.** The package is unreleased; renames of exported names are free.

**D3 (convention) Drop the `irrep`/`ITO` prefixes everywhere** (files, types, functions, exports, docs).
Names: `IrrepOperator` → `Letter`, `ITOKey` → `Factor`, `ITOTermTable` → `TermTable`, `ITOGraph` → `SweepGraph`,
`irrep_mpo_tensors` → `mpo_tensors`; the `IrrepTensorOperators` submodule is dissolved.

**D4 Initial reorganisation before chunk 1** (moves separate from renames), covering O1, O3, O4, O5, O6, O8:
drop `src/operators/`; merge `memo.jl` into `linalg.jl` and `connectedcomponents.jl` into `bipartite.jl` (both in
`utility/`); `irrepinterning.jl` into the term table; split `irrepalgebra.jl` (terms / couple / instantiate) and
move `canonicalize!` next to `Terms`; `builders.jl` → `adjoint.jl`; `expterms.jl` out of `infinite/`; output
files into `mpo/`; tests mirror `src/`; `research/` → `benchmark/n2/`. `kronpower.jl` stays in `timeevolution/`
(only `taylor.jl` uses it). O7 (algorithm selectors) is deferred to chunk 1.

**D5 (process) Verification cadence.** During a chunk: only the test files mirroring the touched `src` files plus
the old-vs-new equivalence script, run through a persistent Revise session (`julia-daemon` skill). Once per chunk:
one full `Pkg.test()` in the background. At the finish: full suite, strict docs build, Julia 1.10 (`min`).

## Possible improvements

## Open items

## Resume here

Setup done; global map presented. Next: user feedback on chunk plan and organisation proposals O1–O8, then chunk 0.
