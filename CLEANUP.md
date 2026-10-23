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
| 0 | Top-down tour of the main path | — | done |
| 1 | Foundations | utility/, compression/algorithms.jl, lattices.jl | done |
| 2 | Letters | letters, siteoperator, instantiate | implementing |
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
As applied: `40b562c` (src moves), `9928b09` (tests mirror src in subfolders; `find_tests` recurses),
`5cf798c` (D3 renames; also `_op_at_ito` → `_factor_at`, `_irrep_channels` → `_sweep_channels`, `_irrep_sweep` →
`_sweep`, `_ito_coefficients` → `_coefficients`), `f263073` (research → benchmark/n2). Full suite 139411/139411,
strict docs build clean (runs every example), 321 `using OpSum:` names all resolve. `algorithms.jl` sits just before
`mpo/symbolicmpo.jl` (needs `BipartiteAlgorithm`).

**D5 (process) Verification cadence.** During a chunk: only the test files mirroring the touched `src` files plus
the old-vs-new equivalence script, run through a persistent Revise session (`julia-daemon` skill). Once per chunk:
one full `Pkg.test()` in the background. At the finish: full suite, strict docs build, Julia 1.10 (`min`).

**D6 Coefficients stay `ComplexF64`** for this review; making them parametric is wanted later: https://github.com/lkdvos/OpSum.jl/issues/43.

**D7 (convention, sharpens D1) Underscore rule.** No `_` on anything called from another file or from a test; `_`
only for helpers used within their own file. Applied chunk by chunk.

**D8 Chunk 1 (foundations).**
(a) O7: one algorithm argument, `mpo(H, alg::BondStrategy = VertexCover())`; `IndependentSVD(trunc)` /
`SequentialSVD(trunc)` used directly; `BipartiteAlgorithm`, `SVDBondAlgorithm`, `bondstrategy` deleted; lossless
default `trunc = notrunc()` (no `nothing`), `T <: TruncationStrategy`; infinite-chain / channel errors by dispatch.
(b) D7 renames: `_cached` → `cached`, `_maxsite` → `maxsite`, `_invariant` → `invariant_violation` (moved to
utility); `_lattice_sectortype` deleted (use `sectortype`); `_tolattice` deleted (`OperatorSum(sites)` →
`FiniteChain(sites)`).
(c) `FiniteChain{B<:Basis}` / `InfiniteChain{B<:Basis}`; migration error text dropped, one-symmetry check kept.
(d) `utility/linalg.jl` → `utility/helpers.jl` (sparse helpers, `cached`, `invariant_violation`).
(e) `min_vertex_cover_bipartite` → `minimum_vertex_cover` returning `(coverU, coverV)`;
`bipartite_connected_components` → `connected_components`.
(f) merge `test_bipartite_components.jl` into `test_bipartite.jl`; drop unused `AbstractAlgorithm` import.
As applied: `8d90b7c` (a; `truncates(strategy)`, `_compress_channels` for channel errors), `fa2d8c7` (b–d;
`OPCACHE_LOCK` shared with letterproducts.jl; `lattices.jl` now included after `algebra/basis.jl`, `_checklattice`
moved to operatorsum.jl; `OperatorSum(::Vector{<:Basis})` only, non-lattice input is a MethodError), `1ed3d91`
(e, f). Equivalence: 23 models / 95 results (default + both SVD sweeps lossless and `truncrank(2)`) bitwise equal to
base. Targeted tests and the two affected examples pass.

**D9 Chunk 2 (letters, SiteOperator, instantiation).**
(a) `Factor(letter, bond)`: drop the always-1 `vertex` field; field `op` → `letter`; vertex 1 only in `_tree_from_bonds`.
(b) Explicit matrix-unit alphabet: `instances(Letter, V)` → `matrixunits(V)`, `instantiate(::Letter|::SiteOperator,
::ElementarySpace)` → `matrixunit(x, V)`; `instantiate` on letters means "in this `Basis`" only.
(c) Canonical `SiteOperator`: letters sorted, no zero coefficients, `+` merges sorted lists; `==`/`hash` are operator
equality; `_canonform` and redundant `prune` calls removed. Any change in stored MPO entries is reported before
keeping it.
(d) `@assert` → `invariant_violation`/`ArgumentError`; `scalartype(::Type{<:Letter}) = ComplexF64` (one method);
delete `isreal(Letter)`; default `==`/`hash` for `Letter`/`Factor`, tuple `isless`; one import block per file.
(e) Dense oracle: `_instantiate_terms` → `dense(h::Terms, sites)` (no mutation of `h`), `_instantiate_term` →
`dense(t::Term, sites)`, `_embed_field` → `embed_letter`; drop it if `_embed_caterpillar` at `K=1` agrees.
(f) `scalarop` and `instantiate(::SiteOperator, …)` into `siteoperator.jl`; `ispassthrough(op) = op ==
passthrough(sectortype(op))`.

## Possible improvements

## Open items

- Tour findings, to decide in their chunks: `mpo(H)` canonicalises `H.terms` in place (7); `TermTable.nvertices` is
  really `N` (7); four-way `_compress` branching + silent `nothing` fallback from the metric path (1/8, with O7);
  `_sweep` = `_sweep_channels(; jordan=false)[1:2]` (7); 35-field `SweepGraph` with positional constructor (7);
  tri-state `start`/`finish` (9); `mpo_tensors(Ws, secs, sites)` / `mpo_terms(Ws, secs)` public raw forms (9);
  `_invariant` defined in graph.jl but used across layers → utility (1); `ComplexF64` everywhere (question).
- Stale refs to removed design notes: `src/infinite/window.jl:3` (research/infinite-mpo.md), `src/algebra/basis.jl:3`
  (research/operator-bases.md).
- `benchmark/n2` includes `MPSKitExperimental.jl/examples/opsum_qchem_su2.jl` (outside this repo), which still uses
  `OpSum.IrrepOperator` → needs `Letter` there; and an out-of-repo include is fragile (chunk 14).
- `scalarop` landed in `algebra/instantiate.jl`; `siteoperator.jl` may be its home (chunk 2).
- `test/utility/test_bipartite_components.jl` could merge into `test_bipartite.jl` (chunk 13).

## Resume here

D1–D5 decided; initial reorg applied. Chunk 0 presented; D6 recorded. Chunk 1 applied; full suite green at 1ed3d91. Chunk 2 decided (D9); implementation delegated.
Equivalence harness: scratchpad/equiv/{equiv,compare}.jl, base result scratchpad/equiv/base.jls, base worktree scratchpad/base.
