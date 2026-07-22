# Flat term-list storage for MPO bond optimization — implementation plan

## Status

Proposal / not started. Written up for handoff to an implementing agent. No code
changes accompany this document.

## Context

This plan is one of several improvement ideas that came out of comparing OpSum.jl's
MPO-construction pipeline against
[ITensorMPOConstruction.jl](https://github.com/ITensor/ITensorMPOConstruction.jl)
(algorithm described in Corbett & Miyake, Phys. Rev. B 112, 16 (2025),
arXiv:2506.07441, building on Ren/Li/Jiang/Shuai, J. Chem. Phys. 153, 084118
(2020), arXiv:2006.02056). See also `research/term-algebra-redesign.md` and
`research/symmetric-operators-implementation-plan.md` for the prior redesign
this builds on.

Four performance-relevant divergences were identified between the two
codebases' per-bond compression step. This document scopes **only the term-list
storage divergence** (item 4 of that list) into an implementation plan. The
other three items are out of scope here and should not be assumed done:

1. Dense `n_left × n_right` coefficient/adjacency matrix per bond in
   `mpo_bond_optimizations` vs. ITensorMPOConstruction's native sparse
   adjacency-list `BipartiteGraph`.
2. No connected-components pre-split before vertex-cover in OpSum, vs.
   `compute_connected_components` in ITensorMPOConstruction, which shrinks the
   vertex-cover problem and exposes embarrassing parallelism.
3. No multithreading anywhere in `graphbuilding.jl`/`bipartite.jl`, vs.
   `Threads.@threads` used throughout ITensorMPOConstruction.
4. **(this document)** Term-list storage: a pointer-chasing, hash-keyed `Trie`
   built once up front, walked at every bond, vs. ITensorMPOConstruction's
   flat, preallocated `OpIDSum` array with all syntactic merging deferred to a
   per-bond sort+scan.

This plan is written to be executable independently of items 1–3, though it
notes where the interface would want to change if those land first (see
"Interaction with items 1–3" below).

## Current architecture (verified by reading the code, not assumed)

- `build_trie!` (`src/operators/globalalgebra.jl:396-443`) walks a `GlobalOp`
  expression tree and, for every term, calls `_emit_leaf!`
  (`globalalgebra.jl:388-394`), which inserts a **dense, length-`N`**
  `site_factors` vector (`N` = number of chain sites/vertices, identity-filled
  at every site the term doesn't touch) as a key into a `Trie{Op,T}`
  (`src/datastructures/trie.jl`). Duplicate terms (identical full-length key)
  are summed at insertion time (`node.value = ... + coeff`).
- `Trie{K,V}` (`trie.jl:1-22`) is a `mutable struct` per node holding a
  `Dictionary{K,Trie}` for its children — insertion/traversal is hash-lookup +
  pointer-chasing, one node per (site, operator) pair per term.
- `mpo_bond_optimizations(vertices, prefix_trie, ::BipartiteAlgorithm)`
  (`src/statemachines/graphbuilding.jl:28-138`) sweeps site by site. At each
  site it re-derives the bipartite grouping **from the trie's live pointer
  structure**: `Us` = children of the current left-node frontier
  (`graphbuilding.jl:49-56`), `Vs` = grouped by full remaining-suffix identity
  via `Dictionary{Vector{Op},Int}` (`graphbuilding.jl:59, 66-70`) — i.e.
  suffix-grouping is recomputed by hashing whole `Vector{Op}` suffixes fresh at
  every bond, not read off a precomputed suffix-sharing structure. It then
  builds a **dense** `zeros(T, length(Us), uid!.current)` coefficient matrix
  (`graphbuilding.jl:73`) before running `min_vertex_cover_bipartite`.
- `mpo_bond_optimizations(..., ::SVDBondAlgorithm)` (`graphbuilding.jl:173-279`)
  already does something closer to what this plan proposes, but only halfway:
  it flattens the trie once via `all_ops = [ops for (ops,_) in
  pairs(prefix_trie)]` (`graphbuilding.jl:182-183`) — so a `Trie` is still
  built and fully walked once — and then computes **per-bond transition IDs**
  via `Dictionary{Tuple{Int,Op},Int}` transition tables (`pre_trans`,
  `suf_trans`, `graphbuilding.jl:197-217`) rather than trie pointer-chasing.
  This transition-ID technique is the right building block to generalize —
  see "Design" below.
- `Trie` is a plain prefix trie, not the `SDAWG`/suffix-DAWG CLAUDE.md
  describes — there is no persistent suffix-sharing structure anywhere in this
  path today. (CLAUDE.md is stale on this point; do not assume `SDAWG` is
  wired into bond optimization.)
- No fermionic operator support exists in `src/` today (`FermionOperator`
  mentioned in CLAUDE.md does not exist; `grep -rn "FermionOperator" src/`
  returns nothing, and there is no fermion-sign/JW-string logic anywhere in
  the current term-insertion or bond-optimization code). This simplifies scope
  now, but flag it: if fermionic operators are added later, the flat
  representation designed here needs a sign-tracking hook analogous to
  ITensorMPOConstruction's `sort_fermion_perm!` (see "Non-goals").
- The ITO track (`src/operators/irreptrie.jl`, `irrepmpo.jl`) is a
  **deliberately separate, byte-for-byte-independent code path** that mirrors
  `BipartiteAlgorithm` but keys the trie on `ITOKey = (op, bond_charge,
  vertex)` instead of bare `Op`, so that trie-node identity already implies
  symmetry-sector purity. This was done on purpose to avoid destabilizing the
  newly-built ITO pipeline while the dense pipeline was still using the older
  representation. This plan preserves that separation (see "Scope boundaries").

## What ITensorMPOConstruction does differently (verified from source)

- `OpIDSum{N,C,Ti}` (`src/OpIDSum.jl`) is a single preallocated flat array:
  `_data::Vector{NTuple{N,OpID{Ti}}}`, reinterpreted as an `N × max_terms`
  matrix, plus a parallel `scalars::Vector{C}` and an
  `num_terms::Threads.Atomic{Int}` counter. `add!` writes one row and
  atomically bumps the counter — no tree, no per-insertion hashing, safe for
  concurrent insertion.
- Critically, **`N` here is the term's maximum operator arity (a bounded
  few-body count), not the number of chain sites.** Each term stores only its
  non-identity `(site, op-id)` pairs (`OpID{id,n}`), padded with a zero
  sentinel — a sparse-per-term encoding. OpSum's current `site_factors` vector
  is dense-per-term (length `N` = number of sites, identity-filled) — for
  local few-body Hamiltonian terms this is asymptotically more storage than
  necessary and is itself worth changing, not just the tree-vs-array question.
- Per-bond merging of syntactically-identical continuations
  (`combine_duplicate_adjacent_right_vertices!`, `BipartiteGraph.jl`) is a
  **sort-then-linear-scan** over contiguous arrays (parallelized with
  `Threads.@threads` for the equality-check pass), not a persistent shared-tree
  structure — merging is deferred entirely to the per-bond step rather than
  being built in at insertion time.

## Goal

Replace the `Trie`-as-mandatory-intermediate representation in the **dense
(Pauli) MPO pipeline's bond optimization** with:

1. A flat, preallocated term-list type (mirroring `OpIDSum`'s sparse
   `(site, op)`-pair-per-term encoding, capped at a maximum term arity) as the
   canonical output of `build_trie!`'s replacement.
2. A per-bond **transition-ID** construction (generalizing the pattern already
   proven in `SVDBondAlgorithm`'s `pre_trans`/`suf_trans`) that produces the
   same `Us`/`Vs`/adjacency information `BipartiteAlgorithm` needs, without
   ever materializing a pointer-based `Trie`.

The output contract of `mpo_bond_optimizations` (`Vector{SparseMatrixDOK{LocalOp{T,Op}}}`)
must not change — this is purely a replacement of the internal representation
feeding it, so no downstream consumer (MPO assembly, existing tests) should
need to change.

## Scope boundaries

- **In scope**: the dense/Pauli pipeline only — `build_trie!`,
  `GlobalOp`/`Sum`/`SiteOp` term ingestion, and
  `mpo_bond_optimizations(..., ::BipartiteAlgorithm)` /
  `mpo_bond_optimizations(..., ::SVDBondAlgorithm)` in `graphbuilding.jl`.
- **Out of scope, explicit follow-up**: the ITO track
  (`irreptrie.jl`/`irrepmpo.jl`). It was deliberately kept independent of the
  dense pipeline's implementation details; do not merge the two code paths in
  this pass. If a flat-storage port to the ITO track is pursued later, the
  per-bond transition key must become `(prev_id, op, bond_charge)` instead of
  `(prev_id, op)`, so that per-bond-sector purity (`ITOKey`'s current
  guarantee, asserted in `irrepmpo.jl`) is preserved exactly, not just
  approximately.
- **Out of scope**: items 1–3 (sparse adjacency-list `BipartiteGraph`,
  connected-components pre-split, multithreading). This plan should compose
  with them later but does not depend on them — Phase 2 below feeds the
  existing dense `min_vertex_cover_bipartite` unchanged.
- **Out of scope**: the QR/rank-decomposition hybrid refinement discussed
  separately; not part of this storage-layer change.

## Design

### 1. Flat term-list type

Introduce a new type, e.g. `TermTable{Op,T}` (naming is the implementing
agent's call), holding:

- A fixed maximum arity `K` (Julia type parameter, like `OpIDSum`'s `N`) —
  determined from the actual maximum number of non-identity factors across all
  input terms, with a floor of 2 to avoid the same degenerate
  `reinterpret(reshape, ...)` edge case `OpIDSum` works around for width 1.
- Per-term storage: `site::NTuple{K,Int}` and `op::NTuple{K,Op}` (or a single
  reinterpreted array as `OpIDSum` does — implementer's choice, but prefer
  whatever keeps the hot per-bond loop allocation-free), padded with a sentinel
  for unused slots (e.g. site `0`).
- A parallel `coeffs::Vector{T}`.
- Coefficient accumulation for exact-duplicate terms: `build_trie!` currently
  relies on trie-node identity to sum duplicate terms at insertion
  (`globalalgebra.jl:392`). The flat replacement needs an equivalent — either
  (a) an insertion-time `Dictionary{NTuple{K,Pair{Int,Op}},Int}` mapping
  canonical sparse term content to row index (same spirit as `OpIDSum`'s
  reliance on downstream dedup, but OpSum's insertion-time dedup is relied on
  by existing tests — check `test_operatoralgebra.jl`/`test_trie.jl` for
  exact-duplicate-term assertions before deciding), or (b) defer dedup to the
  first bond-optimization sweep (duplicates become distinct rows that get
  merged by the same per-bond mechanism that merges any other
  syntactically-identical continuation — verify this actually produces
  identical results to today's insertion-time sum before relying on it).
- Constructed directly from `GlobalOp`/`Sum`/`SiteOp` (mirroring
  `build_trie!`'s traversal in `globalalgebra.jl:396-443`), skipping the `Trie`
  entirely — i.e. this replaces `build_trie!`'s role, not just its output type.

### 2. Per-bond transition-ID sweep (generalizes `SVDBondAlgorithm`'s existing pattern)

`graphbuilding.jl:194-217` already computes, for `BipartiteAlgorithm`'s sibling
algorithm, `pre_ids[t,b]`/`suf_ids[t,b]` via incremental
`Dictionary{Tuple{Int,Op},Int}` transition tables — assigning an integer ID to
each distinct (previous-ID, local-op) pair seen so far, per bond, without ever
walking a pointer tree. Generalize this same construction to run directly off
the new flat `TermTable` (not off `pairs(prefix_trie)`, which still requires a
fully materialized trie today), and use the resulting per-bond ID assignment
to build the `Us`/`Vs` grouping `BipartiteAlgorithm` needs — i.e., produce
something equivalent to today's `Us::Vector{Trie}`/`Vs::Dictionary{Vector{Op},Int}`
(`graphbuilding.jl:37-78`), but sourced from `pre_ids`/`suf_ids` integer arrays
instead of trie nodes. Feed this into the existing (unchanged)
`min_vertex_cover_bipartite`.

Net effect: `BipartiteAlgorithm` and `SVDBondAlgorithm` converge on sharing the
same flat-storage-plus-transition-ID front end; only the final compression
step (vertex-cover vs. SVD) differs between them, rather than duplicating two
separate representations of "what terms look like at this bond" as happens
today.

### Interaction with items 1–3 (informational only, not required for this plan)

If the sparse `BipartiteGraph`-style adjacency-list rewrite (item 2 from the
comparison) lands later, its natural input is exactly the `Us`/`Vs` +
`pre_ids`/`suf_ids` structure this plan produces — so doing this plan first is
a reasonable prerequisite, not wasted work, if item 2 is picked up next.
Connected-components pre-splitting (item 3) and multithreading (item 4 in that
numbering — unrelated to this document's "item 4") would slot in between the
transition-ID construction and the vertex-cover call.

## Phased rollout

- **Phase 0 — Investigation spike.** Before writing any code: read
  `test_trie.jl`, `test_operatoralgebra.jl`, `test_mpo_bipartite.jl` to confirm
  (a) whether any test relies on insertion-time exact-duplicate-term summation
  specifically (vs. summation happening anywhere in the pipeline), (b) whether
  `Trie`/`build_trie!` are used or exposed anywhere outside
  `graphbuilding.jl`'s two `mpo_bond_optimizations` methods (public API
  surface, docs, other call sites), and (c) the actual distribution of term
  arity vs. chain length in the existing test Hamiltonians, to sanity-check
  the sparse-per-term storage assumption is worthwhile. Do not proceed to
  Phase 1 until these are answered.
- **Phase 1 — New flat type, no bond optimization yet.** Implement
  `TermTable` (or chosen name) and its construction from `GlobalOp`, with unit
  tests asserting it contains the same term content (coefficients, sites,
  operators) as today's `Trie`-based `build_trie!` output, for a range of
  hand-built and randomly generated `GlobalOp` expressions. No changes to
  `mpo_bond_optimizations` yet.
- **Phase 2 — New bond-optimization front end.** Implement the transition-ID
  sweep (generalizing `SVDBondAlgorithm`'s `pre_trans`/`suf_trans` pattern) and
  wire it into a new dispatch of `mpo_bond_optimizations` that accepts
  `TermTable` instead of `Trie`. Keep the existing `Trie`-based method
  untouched and as the default — this is purely an additive new code path
  during validation. Add equivalence tests comparing the new path's output
  against the old path's output (`≈` on the assembled dense operator, per the
  existing `instantiate`-based oracle tests) across the same random-Hamiltonian
  fuzz set used in Phase 1.
- **Phase 3 — Benchmark before flipping the default.** Measure construction
  time and peak memory of old (`Trie`) vs. new (`TermTable`) paths across
  scaling in term count and chain length `N`, including at least one
  long-range/all-to-all term set (the case most likely to expose the
  pointer-chasing/hashing overhead). Only switch the default
  `mpo_bond_optimizations` dispatch once this shows a real improvement —
  do not assume the win without measuring it.
- **Phase 4 — Follow-up (separate PR, not this plan's scope): ITO port.**
  Evaluate porting the same flat-storage approach to `irreptrie.jl`/`irrepmpo.jl`,
  extending the transition key to `(prev_id, op, bond_charge)`. This needs its
  own validation pass against the ITO track's existing exact-reconstruction
  tests (`test_irrep_trie.jl`, `test_irrep_mpo.jl`, `test_irrep_mpo_tensors.jl`)
  and should not be started until Phase 3 has landed and been stable for the
  dense pipeline.
- **Phase 5 — Cleanup.** Once the new path is default and validated (and the
  ITO port, if pursued, is also done), remove `build_trie!`'s use in
  `mpo_bond_optimizations` and evaluate whether `Trie`/`build_trie!` still have
  any remaining callers; if not, consider whether to keep `Trie` as a
  general-purpose data structure (it may still be useful elsewhere) or remove
  it. Do not remove `Trie` itself as part of this plan — only its mandatory
  role in this specific pipeline.

## Testing strategy

- Reuse `test/test_trie.jl`, `test/test_operatoralgebra.jl`,
  `test/test_mpo_bipartite.jl` as the correctness oracle throughout — the new
  path must reproduce identical (`≈`) assembled operators for every existing
  test Hamiltonian.
- Add property-based/fuzz equivalence tests: generate random `GlobalOp`
  expressions (random term count, site count, operator arity, coefficient
  values) and assert old-path and new-path `mpo_bond_optimizations` outputs
  reconstruct to the same dense operator via `instantiate`.
- Add a dedicated test asserting duplicate-term coefficient accumulation still
  works correctly under whichever dedup strategy Phase 1 settles on (insertion-time
  vs. deferred-to-bond-sweep).

## Explicit open questions for the implementing agent

1. Exact-duplicate-term dedup strategy (insertion-time dictionary vs. deferred
   to bond sweep) — resolve via Phase 0 investigation, not by assumption.
2. Fixed-width `NTuple`-based row storage (mirrors `OpIDSum`, likely better for
   type stability and cache locality) vs. `Vector`-of-`Vector` per term —
   recommend fixed-width, but confirm no existing use case needs unbounded
   term arity.
3. Whether `Trie`/`build_trie!` have call sites or public-API surface beyond
   `graphbuilding.jl` that this plan would otherwise break.
4. Naming of the new type and its constructor functions — not specified here,
   left to the implementing agent to match existing repo conventions (see
   `src/operators/globalalgebra.jl`, `src/datastructures/` for naming style).

## Non-goals

- No fermionic sign-tracking is implemented as part of this plan, since no
  fermionic operator type exists in `src/` yet. If `FermionOperator` lands
  before this plan is executed, revisit this document — the flat storage
  layout should probably grow a sign-tracking hook analogous to
  ITensorMPOConstruction's `sort_fermion_perm!` at that point, but designing
  that now would be speculative.
- No change to the ITO track in this pass (see "Scope boundaries").
- No change to `min_vertex_cover_bipartite`'s dense-matrix internals (item 1
  from the comparison) — this plan feeds it unchanged.
- No multithreading added (item 3 from the comparison).
