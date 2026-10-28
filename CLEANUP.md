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

Refreshed in chunk 12 (D19), with the include order of `src/OpSum.jl` after D19(d); arrows = "is used by".

```
utility/ (helpers: invariant_violation, memo lock; bipartite: Hopcroft–Karp, König, components)
  ──► algebra/  letters (Letter, Factor) → siteoperator (SiteOperator, scalarop) → terms (Term, Terms, PendingTerm)
                → recouple → couple (couple, dot) → instantiate → localoperator (LocalOperator)
                → basis (Basis: duals, metric, letter products) → projection (project, project_units)
  ──► lattices (AbstractLattice, FiniteChain, InfiniteChain)
  ──► algebra/  adjoint (adjoint_terms) → products (resolve) → letterproducts (letter_product, site_product)
                → stockbases (spin_basis …, spin_ops …; memoised)
  ──► expterms (expterm, ExpSum, ExpKey, ExpChannel) ──► operatorsum (OperatorSum, opsum, checkinsert, checkgenerator)
  ──► compression/ termtable (TermTable, suffix/prefix ids) → strategies (BondStrategy, VertexCover, IndependentSVD,
                   SequentialSVD, SVD_ATOL) → channels (ChannelState automaton) → graph (SweepGraph, sweep, at_site!,
                   BondBasis) → graph_vc (ComponentCover) | graph_svd → metric (compress_in_metric)
  ──► infinite/ unitcell (translate, unitcell_terms, window_terms) → window (infinite_window)
  ──► mpo/      symbolicmpo (SymbolicMPO, mpo, mpo_terms, islossless) → tensors (mpo_tensors, mpo_tensormap)
                → jordan (JordanLayout, jordan_layout, jordan_mpo_tensors)
  ──► timeevolution/ kronpower (mpo_power, KronSite) → taylormpo (TaylorEntry, TaylorMPO, evaluate)
                     → taylor (TaylorWork, the passes, taylor_mpo, run_passes)
ext/OpSumMPSKitExt (instantiate → MPSKit MPOs, TaylorCluster)
```

Main path: `opsum(lat, ops...)` → `OperatorSum` (`checkinsert`, `resolve`) → `mpo(H, alg)` (symbolicmpo.jl) →
finite: `_compress` → `TermTable` → `sweep(tt, N, alg; channels)` → per site `at_site!` → `bond_basis!` (VC / SVD,
`BondBasis`) → `SymbolicMPO(lat, Ws, bondsectors, start, finish)`; infinite: `unitcell_terms` → `infinite_window` (same
sweep, `jordan = true`, fixed-point cell) → `mpo_tensors` / `jordan_mpo_tensors` / `instantiate` (ext) /
`taylor_mpo` (`mpo_power` → passes → `TaylorMPO` → `evaluate`).

Core types: `Letter` · `Factor` (letter + outgoing bond charge) · `SiteOperator` (sparse sum of letters) · `Term`/`Terms`/
`PendingTerm` · `LocalOperator` · `Basis` · `FiniteChain`/`InfiniteChain` · `ExpSum` · `OperatorSum` · `TermTable` ·
`SweepGraph` · `BondStrategy` · `SymbolicMPO` · `JordanLayout` · `TaylorEntry`/`TaylorMPO`/`TaylorWork`.

Global mutable state: memo caches `_STOCK_CACHE`, `_ADJOINT_CACHE`, `_RECOUPLE_CACHE`, `_PRODUCT_CACHE` (`Dict{Any,Any}`)
under `OPCACHE_LOCK`; letter products are memoised per `Basis`. Never cleared.

Named tolerances: `BASIS_RTOL`, `METRIC_ZERO_RTOL`, `METRIC_TRIVIAL_ATOL`, `RECOUPLE_ATOL`, `PRODUCT_RTOL`, `SVD_ATOL`;
window sizes `WINDOW_EDGE_MARGIN`, `WINDOW_CHECK_CELLS`, `WINDOW_DOUBLINGS`.

Sizes (lines, at 0e811bc): src 5.9k (algebra 2.3k, compression 1.4k, timeevolution 0.8k, mpo 0.5k, infinite 0.2k,
top level 0.7k), test 6.4k. Largest: compression/graph.jl 552, timeevolution/taylor.jl 430, algebra/terms.jl 382.

## Chunks

| # | Chunk | Files | Status |
|---|-------|-------|--------|
| 0 | Top-down tour of the main path | — | done |
| 1 | Foundations | utility/, compression/algorithms.jl, lattices.jl | done |
| 2 | Letters | letters, siteoperator, instantiate | done (full suite green at 4b54d6f) |
| 3 | Terms | terms, couple | done (full suite green at bf826f5) |
| 4 | Bases and projection | basis, projection, stockbases, adjoint | done (full suite green at f9990dc) |
| 5 | Unplaced operators and products | localoperator, products, letterproducts | done (full suite green at a53c9c4) |
| 6 | Container and channels | operatorsum, expterms | done (full suite green at 9bccca7) |
| 7 | The sweep | compression/termtable, graph | done (full suite green at ea13cfe, 139814) |
| 8 | Bond bases | compression/graph_vc, graph_svd, metric | done (full suite green at ca2ce05, 139827) |
| 9 | Output | mpo/symbolicmpo, tensors, jordan, ext/ | done (full suite green at ca2ce05, 139827) |
| 10 | Infinite chains | infinite/unitcell, window | done (full suite green at dc5e5ce, 139829) |
| 11 | Time evolution | timeevolution/ | done (full suite green at 59da1aa, 139835); (e) option B at 9d641f0, targeted tests |
| 12 | Organisation revisit | — | done (full suite green at 911f5ce, 139883) |
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
As applied: `7ea21d0` (a), `ca8d3a0` (b, f; `scalarop(c, ::ElementarySpace)` deleted; file-local `_space` kept for
the dense oracle on bare spaces; this commit alone breaks that oracle, fixed in `0907daf`), `72fd2f7` (c, alone),
`0907daf` (d, e; `_embed_caterpillar` at K=1 equals `embed_letter` to 1.5e-14 on 450 cases, so `dense` has no K=1
case; `embed_letter` kept for products.jl), `4b54d6f` (field-wise `hash` for `Letter`/`Factor` restored: the default
struct hash made `mpo` 25–30% slower on product-sector models; interleaved timings now within 2–4% of `ca8d3a0`,
scripts in scratchpad/perf). Equivalence: 6/95 cases differ bitwise, all operator-identical (exact-zero letters no
longer stored in 17 entries; letters sorted); same stored entries and bond sectors. `SiteOperator` `+` 25% faster.

**D10 Chunk 3 (terms and coupling).**
(a) `Terms` is canonical at all times: every producer returns normal form (`+` merges sorted lists, constructors and
`opsum` canonicalise once); pending products kept apart (no normal form, as now). Reading never mutates;
`canonicalize!` becomes internal; `nterms_raw` removed; `mpo(H)` no longer touches `H`. Verify with the
equivalence harness and a construction-time check (largest showcase model, N₂ terms).
(b) `Term` `==`/`hash` include the coefficient; canonicalisation keys on an internal `support(t)`.
(c) Renames: `Term.keys` → `factors`, `ops(t)` → `letters(t)`; D7: `tree_from_bonds`, `isfermionodd`,
`canreorder`, `couple_terms`.
(d) `Term` checks → `ArgumentError`; `via` keyword deleted; one GenericFusion entry check for both `couple` forms
(removes the `@assert`); `_spaceaslattice` deleted.
(e) Latticeless `opsum(args...)` → `Terms(args...)`; `opsum` always builds an `OperatorSum`; `append!` stays.
As applied: a718882 (c), 7c6e751 (b, d), 2040882 (a), 2897433 (e), 04e2747 (tests keyed by `support`, `TermBag`
with `_written!`; commits 7c6e751/2040882 alone fail 3 test files, fixed there). New internal `TermBag` (raw
accumulator) so no code path writes a non-canonical `Terms`; `≈` is a merge walk on sorted lists; `Terms(Term{I}[])` is
the empty operator. `Term`-keyed dictionaries in `unitcell.jl`/`expterms.jl` now key on `support` (the unit-cell
translate check needed it). Equivalence: 0 bitwise diffs vs pre-chunk (95 cases), 0 operator-level vs base. Build of
`H` up to +24% (hubbard_su2 N=128 0.19→0.23 ms; haldane_shastry N=256 20→22.5 ms), `mpo(H)` unchanged within noise.
`benchmark/n2` was already stale (`_rebasis` signature), chunk 14.

**D11 Chunk 4 (bases, projection, stock bases, adjoint).**
(a) `Basis` equality is identity (`==` is `===`, `hash` by `objectid`), documented; stock bases memoised so repeated
calls are identical.
(b) Bases are always complete: `complete` keyword removed, `Basis(ops)` completes with matrix units.
(c) `project` = overlaps with precomputed dual elements `ẽ_k = Σ_m (G⁻¹)_km e_m` (coefficient `⟨Ẽ, h⟩ / g` per
letter tuple and tree); `_rebasis`, the `pinv` chop and the per-projection residual check go (residual stays in
tests); matrix units only used inside basis construction. Equivalence harness decides round-off handling (chop on
`⟨Ẽ, h⟩` at the old tolerance if tiny letters appear).
(d) `Basis` owns all basis-dependent precomputed data: dual elements, the letter metric (now `_lettermetric` in
metric.jl), the letter product table (now the global `IdDict` in letterproducts.jl, filled lazily per basis), and
anything per-basis the adjoint needs; global per-basis caches go.
(e) Narrow `_tkt_operator`'s `catch`; files split by concern (basis.jl = construction + instantiate,
projection.jl = `project`); header fixed; D7 renames; adjoint and stock caches kept, documented.
(f) Issue: `project` builds each candidate densely; overlaps factorise site by site. → #44.
As applied: 1fed91e (one commit). `Basis` is a `mutable struct` (identity `==`/`hash`) with fields `duals`,
`norms`, `metric` (eager), `products` (lazy, under `OPCACHE_LOCK`); `_LETTER_PRODUCT_CACHE` gone, `_ADJOINT_CACHE` kept
(keyed per term × lattice). Dual of `e_k` is `Σ_m conj((G⁻¹)_km) e_m` (conjugate needed for complex bases; tested
`⟨ẽ_k, e_j⟩ = δ`); `G⁻¹` chopped at `rtol` scale-invariantly. Completeness checked per charge at construction
(`ArgumentError`). `_tkt_operator` rethrows all but the "not symmetric under" `ArgumentError`. Equivalence: VC cases
identical up to round-off (max 1.8e-15 abs, same letters/entries); SVD cases differ by noise and degenerate gauge,
lossless SVD still reproduces the operator (11 cases checked); 2 trunc2 cases differ by tie-breaking in one bond.
Residuals ≤ 7.5e-15. `project` 1.4–2.6× faster, ~half the allocations; model construction ~2× faster (recovers the
chunk 3 build cost); `mpo` unchanged. Lattice letter-range check is weaker (complete same-space bases have equal
letter counts). `benchmark/n2` got a local `rebasis` helper, still unrunnable (chunk 14).

**D12 Chunk 5 (unplaced operators, products, letter products).**
(a) P5.1: new `algebra/recouple.jl` (included before `couple.jl`) holds re-coupling (`recouple`, `concat`, helpers);
`products.jl` keeps placed `*` and pending-product resolution.
(b) P5.2: re-couple through TensorKit's fusion-tree operations instead of dense charge-only tensors (`_treetensor`,
`_expand`, the `1e-13` cutoff go); fall back to the current route only if the API cannot express it.
(c) P5.3: drop `_RECOUPLE_CACHE` if (b) makes re-coupling cheap (timed); `_PRODUCT_CACHE` stays.
(d) P5.4: named, documented tolerances instead of bare literals. P5.5: D7 renames.
(e) Scalars in coupling are not supported: unplaced `couple`/`dot` reject an operand term without a charged letter in
its slot, like the placed form, so `couple(ops...)[sites] ≡ couple((op[site] for …)...)` including errors. The
identity stays available as an operator: `one(SiteOperator{I})`/`scalarop(c, B)`, `one(::Terms)`, a `LocalOperator`
identity, `B ± α`; documented together.
(f) P5.6: issue for per-fusion-channel structure constants (unify on-site products, non-abelian `taylor_mpo`). → #45.
As applied: edc25ba (a, renames), 0c71220 (b), 124743e (d), 1c0e109 (e), 18ef149 (identity docs, `one(::LocalOperator)`,
`VectorInterface.add` for `SiteOperator`, removes the long-standing warning). Re-coupling uses TensorKit 0.17.2
`permute(::FusionTreeBlock, …)` (plain tree `permute` asserts `UniqueFusion`) and `merge`; `FusionTreeBlock` is
semi-internal TensorKit API. `RECOUPLE_ATOL = 1e-13` stays (structural zeros come out as ~1e-17 noise and would change
the cover); `BASIS_RTOL`, `PRODUCT_RTOL` named. `_RECOUPLE_CACHE` kept: dropping it makes re-coupling workloads 1.3–2.2×
slower hot. Equivalence: 95/95 bitwise identical to pre-chunk; new vs old re-coupling ≤4.4e-16 on 3600 random cases,
exact on all 100 cases the tests hit. `mpo` and model construction unchanged. `test/algebra/test_recouple.jl` new.

**D13 Chunk 6 (OperatorSum and decaying channels).**
(a) P6.1: one collector: `opsum!` reuses `collect_terms!` (extended with a channel slot), so `Terms(…)` and `opsum`
accept and reject the same inputs (numbers, unplaced operators, `OperatorSum`s rejected by both).
(b) P6.2: arithmetic in each type's own file (scaling/`-` for Term/Terms → terms.jl, LocalOperator → localoperator.jl);
`≈` on `ExpSum` in expterms.jl; `ExpSum` accumulation helper so `opsum!` stops writing `.channels.channels`.
(c) P6.3: exact `==`/`hash` on `OperatorSum` (lattice, terms, channels) and `ExpSum`.
(d) P6.4: `ExpKey(support, exitsite, decay, string::SiteOperator{I})` (no `Term`, no dead coefficient, no
`Union{Nothing}`); `isless` on `ExpKey` replaces `_channelorder`; `@assert` → `invariant_violation`.
(e) P6.5: automaton (`ExpChannel`, `ChannelState`, `lower_channels`, `lastentry`, `isentry`) → `compression/channels.jl`;
descriptor, `expterm` and the expansion oracle stay in `expterms.jl`.
(f) P6.6: `merge_terms!` shortcut when the target is empty, kept only if it measurably helps building `H`.
(g) P6.7: D7 renames (`lower_channels`, `checkinsert`, `checkchannels`, `checklattice`, `checkgenerator`, `resolve`
as applicable); readable `_checkchannels` errors.
As applied: 4765a1a (e, pure move), ed9d17d (b, arithmetic), 1f493a4 (a, c, d, g, `ExpSum` helpers), 5d5be8d.
`TermBag{I,C}` carries a channel slot (`nothing` for `Terms`); one `rejection(a)` hook per type, so `Terms(…)` and
`opsum` give identical errors. `addchannel!`/`addchannels!`, `copy(::ExpSum)`. Exact `==`/`hash` on `OperatorSum`,
`ExpSum`, `ExpKey`, `Terms`, lattices. (f) tried and reverted: ≤1% on building `H`. Equivalence: 95/95 bitwise identical
(bond order included). `mpo` and channel-model timings unchanged or slightly better.

**D14 Chunk 7 (TermTable and the sweep graph).**
(a) P7.1: `TermTable(ts)` without `nvertices`; `N` passed where needed.
(b) P7.2: one entry point `sweep(tt, N, strategy; jordan, channels)` returning `(Ws, bondsectors, starts, finishes)` as
a named result; `IndependentSVD` keeps its own method; `_sweep`/`_sweep_channels` go.
(c) P7.3: `SweepGraph` fields grouped in small immutable structs (right vertices, left vertices, pending terms,
boundary markers, scratch), keyword construction, `K` dropped; kept only if the perf harness shows no slowdown
beyond noise (else flat struct with keyword construction).
(d) P7.4: `@enum`s (not integer constants) for the right-vertex kind (term class / channel state / exhausted) and the
`_rdesc` kind; the channel index stays a separate integer.
(e) P7.5: lazy insertion for `SequentialSVD` too; if bitwise identical, eager mode, `lazy` and the seeding branch go.
(f) P7.6: D7 renames for the cross-file sweep names; fix the garbled `_iscyclic` comment and "an `TermTable`".
As applied: 47fa34d (a+b; `_compress_channels` and `window.jl` call `sweep(...; jordan = false)`), d7fa9ec (f; file-local
`_prepare_bond!`, `_build_next_graph!`, `_force_finish!`, `_forced_cover`, `_canonicalise_*` keep `_`), 3b8e1b0 (c, keyword
construction only), 7bb5e55 (d; `RightKind` with its own `rkind` column, `DescKind`), 690482f (CLAUDE.md). 95/95 bitwise vs
pre7.jls, 0 operator mismatches. (c) grouping implemented (8f7fdc6, branch `cleanup-grouped-sweepgraph`, 95/95 bitwise,
time within noise) reverted for +16–368 bytes per `mpo`, then taken back by the user's call (1f2e5ac). (e) not possible: the SVD forwards
the lazy sentinel as a dense column (BoundsError in `_build_next_graph!`) and has no start index to inject pending terms;
eager mode stays, documented at `_graph_lazy`. The `rkind` column costs ~0.4% allocation (+2–236 KB per `mpo`); mpo time
+0–3% (seq_hubbard +5.5% min, median within noise); TTFX unchanged (4.6–4.8 s).

**D15 Chunk 8 (bond bases).**
(a) P8.1: SVD routes drop entries below a named `SVD_ATOL` relative to the bond's largest singular value (~1e-14),
in both `SequentialSVD` (`U`, `R`) and `IndependentSVD` (`U'·C·U`); not bitwise, checked at operator level.
(b) P8.2: one metric path; `compress_in_metric` never returns `nothing`, the fallback branch in `_compress` goes.
(c) P8.3: `lettermetric` moves to `algebra/basis.jl`, its `1e-13` tolerances named.
(d) P8.4: named results for `_vc_component` and `_bond_basis!` (a `BondBasis` struct incl. start/finish indices);
strategies no longer write into the graph as a side channel.
(e) P8.5: `_svd_flatten_block` split into `U` and `R` parts; `SequentialSVD` fills `C` directly, no dense `coeff`.
(f) P8.6: `vlocal` for the finish-vertex lookup; `IndependentSVD`/`SequentialSVD` docstrings say they agree in bond
dimension, not gauge; document that `NoTruncation` SVD truncates at `eps`.
(g) `IndependentSVD` is kept.
As applied: 5e504d7 + 1a6c7b7 (c), ed6482b (b), 6db9c93 (f), 522404f (d; `BondBasis`, `ComponentCover`), f67cc18 (e),
1ca3a5c + 5024f49 (a), 68ec223 (CLAUDE.md). Bitwise 95/95 at f67cc18 vs df25411 (= pre7). After (a): VertexCover 23/23
bitwise; lossless SVD ≤7.3e-15 relative to the reference; truncating unchanged to 2.9e-16; stored entries IndependentSVD
4614 → 3450, SequentialSVD 7369 → 4326 (lossless); one case (hubbard_ly4 N=32 seq lossless) has bond sum 1493 → 1496
(near-eps rank decisions). Scale deviation: `SequentialSVD` cuts `U` at absolute `SVD_ATOL` (isometry) and `R` at
`SVD_ATOL·‖S‖∞`; `IndependentSVD` cuts per letter block at `SVD_ATOL·max|C_op|` (`C_op` is a 0/1 structure matrix
except at site N, so a singular value is the wrong scale there); documented at `SVD_ATOL`. Perf vs df25411: VertexCover
−2.6…+1.6%, ≤288 B; SequentialSVD −15…−37%; IndependentSVD +3.7…+7.1% (threshold pass). TTFX unchanged (4.79 s).
Targeted tests green at 1ae8bab. Full suite to run with chunk 9.

**D16 Chunk 9 (output: SymbolicMPO, tensors, Jordan form, MPSKit extension).**
(a) P9.1: finite chains with channels sweep with `jordan = true` if that stays lossless and satisfies the Jordan
invariants; then `start`/`finish` are always length `N`, the "unknown" state and its errors go, and `instantiate`
works with `expterm`s on finite chains. If it does not work, keep the empty-vector state and record why.
(b) P9.2: the sweep's result uses `start`/`finish`; `mpo` constructs `SymbolicMPO` by name, not by splatting.
(c) P9.3: public output takes `SymbolicMPO` only (`mpo_terms(W)`, `mpo_tensors(W)`); the raw `(Ws, secs, …)` forms are
unexported internal helpers (used by `window.jl`, `taylormpo.jl`, no fake `SymbolicMPO` in `TaylorMPO`); tests and docs
use `mpo_terms(W)`; the infinite `mpo_tensors` moves to `tensors.jl`. `mpo_terms` on an infinite chain stays undefined.
(d) P9.4: the extension adds `instantiate` methods directly; `_mpskit_mpo` goes; a `MethodError` hint says to load MPSKit.
(e) P9.5: `closed`/`_close_right` move to `timeevolution/`; `jordan_layout` loses its underscore (D7).
(f) P9.6: concretely typed caches in `_mpo_tensors` (no function barriers), type-stable `blk` accumulator in
`_jordan_tensors`; kept only if time and allocations are not worse.
As applied: c73cc1c (b), 30199f0 + 58eb796 (c; internal `reduced_terms`/`reduced_tensors`), d55a2d3 (d; `instantiate`
docstring on `function instantiate end`, hint tested in test/mpo/test_symbolicmpo.jl), 4dcfb12 (e; `close_right(layout)`
in kronpower.jl), ca91e9b (f), d03630d (a), d5aed57 (CLAUDE.md). (a) works: 34 finite-channel models, bond dims identical
(zero cost), lossless, Jordan form and MPSKit `instantiate` agree with the dense operator; side effect: `taylor_mpo` now
works on finite chains with `expterm` (checked to order N+1). 95 cases: bitwise up to ca91e9b; after (a) only
`hand/finite_expterm` differs (start/finish filled, Ws identical). `mpo_tensors` 1–11% faster, ≤ allocations; `mpo`
±1.3%; TTFX first `mpo_tensors` 7.21 → 5.84 s. Targeted tests (19 files) and docs build green.

**D17 Chunk 10 (infinite chains).**
(a) P10.1: the window reads `start`/`finish` of the fixed-point cell from the sweep (after D16(a)); `_identity_links`,
`_closed_chains`, `_identity_channels` go, kept only if the cells are bitwise identical (same `W`s and indices).
(b) P10.2: remaining internal failures throw via `invariant_violation`; `done` → `finish`; stale research-note ref goes.
(c) P10.3: `contract_open` goes (tests use `window_operator`); `mpo_terms_window` moves to `test/testutils.jl`.
(d) P10.4: `_infinite_window` → `infinite_window`; `_entriesequal` → `==` if equivalent, else `entriesequal`.
(e) P10.5: the `ExpSum` translates error names the clashing generator; `window_terms(ts, L, ncells)`; the window-size
numbers become named constants with their reason.
Implemented by a separate agent after chunk 9.
As applied: a1193c1 (a+b; window sweeps with `jordan = true`, the backbone search is deleted), 1500245 (c), efb3c77 (d;
`==` differs from `_entriesequal` only in treating a stored-but-empty entry as absent), 0ab7c59 (e; `WINDOW_EDGE_MARGIN`,
`WINDOW_CHECK_CELLS`, `WINDOW_DOUBLINGS`). 95/95 bitwise vs pre10.jls; ncells independence tests pass; infinite `mpo`
2–5% faster, 3–7 KB fewer allocations. Targeted tests green at 0ab7c59. The `Terms` translates error still reports the
earlier term's shifted support.

**D18 Chunk 11 (time evolution).**
(a) P11.1 (D7): `_mpo_power` → `mpo_power`, `_taylor_work` → `taylor_work`, `_taylor_mpo` → `run_passes`, `_extend!` →
`extend!`, `_fold_passive!` → `fold_passive!`, `_merge_equivalent!` → `merge_equivalent!`, `_compress!` →
`compress_levels!`, `_prune_orphans!` → `prune_orphans!`, `_scaled` → `scaled`, `_walk!` → `walk_words!`, `_rowbond` →
`rowbond`, `_bond_space`/`_deg_indices` → `bond_space`/`deg_indices`.
(b) P11.2: `taylor_mpo` and the pass runner move to `taylor.jl` (move-only commit); `taylormpo.jl` keeps the types and
`evaluate`.
(c) P11.3: `_kron_entry!` moves to `test_taylor.jl`; the `taylor.jl` header stops citing it; the `TaylorEntry`
docstring drops `_scaled`.
(d) P11.4: `TaylorMPO` gets `iterate`, `eltype`, `firstindex`, `lastindex`.
(e) P11.5: build the power over the levels that matter instead of all `n^N` (fixed now, not an issue): measure first
where `taylor_mpo` spends its time and how many of the `n^N` levels survive; implement only what keeps the final
`TaylorMPO` identical (bitwise, or to round-off if entries get summed in another order) for every pass sequence.
As applied: e8dda51 (a), 163ad1b (b, move only), 628f8c5 (c+d; `_kron_entry!` → `kron_entry!` in test_taylor.jl).
160/160 Taylor records bitwise vs pre11.jls (new harness `scratchpad/equiv/taylor_equiv.jl`, `compare_taylor.jl`);
targeted tests, runic, names, strict docs green. (e) measured, not implemented: level enumeration/index/sectors are
≤2% of `taylor_mpo` (extension on: `extend!` 70–80%, merge 15–19%, power 2–3%); 21–48% of the `n^N` levels are
unreachable, but the reachable set must be closed under the extension and the left-merge classes. Cheapest real win:
`extend!` skipping rows the right-merge discards (~8–22% at N = 3–4; oracle bound 14–45%). Numbers: scratchpad/p115/.
Then option B (user decision): 9d641f0, `extend!(w; skip_merged = true)` from `taylor_mpo` only (default `false`
keeps arbitrary `run_passes` sequences exact), predicate shared with the merge (`_merge_class`). 160/160 bitwise vs
pre11.jls; `taylor_mpo` −8…−23% time, −10…−24% allocations (XXZ N=4 0.427 → 0.357 s, t-V N=4 0.258 → 0.198 s).
The full redesign (skip every row that ends up unused, 14–45%) is left as an issue.

**D19 Chunk 12 (organisation revisit).**
(a) P12.1: `_compress`/`_compress_channels`, `mpo_terms`/`reduced_terms` and `islossless` move from `mpo/tensors.jl` to
`mpo/symbolicmpo.jl` (next to `mpo`); `tensors.jl` keeps the dense assembly (move only).
(b) P12.2: `checkgenerator`/`_checkneutral` move from `infinite/unitcell.jl` to `operatorsum.jl` (next to `checkinsert`).
(c) P12.3: `compression/algorithms.jl` → `compression/strategies.jl`.
(d) P12.4: include list grouped by folder: utility → algebra (letters … projection) → lattices → algebra (adjoint,
products, letterproducts, stockbases) → expterms → operatorsum → compression (termtable, strategies, channels, graph,
graph_vc, graph_svd, metric) → infinite → mpo → timeevolution.
(e) P12.5: the global map and open items here are refreshed (by me, not the agent).
(f) P12.6: `compression/graph.jl` stays one file.
As applied: f56ae51 (a), 47f2435 (b; `_checkneutral` only called by `checkgenerator`, keeps its underscore), 1a0f709
(c), ede617f (d; no deviation from the planned order), 64c386e (CLAUDE.md), 136e7d5 (e). 95 MPO records bitwise vs
pre10.jls, 160 Taylor records bitwise vs pre11.jls; targeted tests, runic, names, strict docs green.

## Possible improvements

## Open items

- `benchmark/n2` includes `MPSKitExperimental.jl/examples/opsum_qchem_su2.jl` (outside this repo), which still uses
  `OpSum.IrrepOperator` → needs `Letter` there; and an out-of-repo include is fragile (chunk 14).
- Tests (chunk 13): no test file for `algebra/adjoint`, `algebra/instantiate`, `compression/channels`,
  `compression/graph_vc`, `infinite/unitcell`, `utility/helpers`; `test_basis_lattice.jl` at top level;
  `test/utility/test_bipartite_components.jl` merge (if still separate); `matrixunit` clash in testutils.
- Issues to open at the end: `Terms` translates error reports the shifted support; IndependentSVD 3.7–7.1% slower
  after the SVD_ATOL cut; hubbard_ly4 N=32 SequentialSVD bond sum +3; Taylor extension skipping every unused row
  (14–45%, D18e); ComplexF64 (#43 exists).

## Resume here

Chunks 0–7 done, full suite green at ea13cfe (139814). Chunk 8 done (D15). Chunk 9 done (D16); full suite green at ca2ce05 (139827). Chunk 10 done (D17), full suite green at dc5e5ce (139829). Chunk 11 done (D18), full suite green at 59da1aa; 9d641f0 targeted only (full suite with chunk 12). Next: chunk 12 (organisation revisit). Chunks 9–14 then finish. Equivalence harness: scratchpad/equiv/{equiv,compare,compare_op}.jl, base result scratchpad/equiv/base.jls;
pre-chunk references pre3.jls … pre7.jls. Full suites run in a throwaway worktree (scratchpad/ft-chunkN).
