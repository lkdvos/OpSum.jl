# Port OpSum.jl's reduced-MPO construction onto ITensor's persistent-graph architecture (non-abelian)

*Handoff prompt for a fresh Claude Code session started in this repo. The research doc it references
(`research/itensor-mpograph-construction.md`) is committed alongside this file.*

## Objective
Re-implement OpSum.jl's symmetry-reduced MPO construction on the **persistent bipartite-graph +
`at_site!` sweep** architecture used by ITensorMPOConstruction.jl, generalized to OpSum's **non-abelian
(TensorKit `Sector`) machinery**. Build it *alongside* the current implementation and prove parity;
do not delete the existing code until the new path passes.

## Required reading (in order)
1. `research/itensor-mpograph-construction.md` — full spec of the target architecture (§3–7), a worked
   example (§8), and a bridge section (§9) mapping every ITensor concept to OpSum's current code. **This
   is your primary spec.** The `at_site!` five-phase description in §6 and the QR/VC backends are what
   you are porting.
2. `src/operators/irreptermtable.jl` — the current sweep `_irrep_bipartite` (transient frontier) and
   `_irrep_svd`; the flat store `ITOTermTable`; `_op_at_ito`, `_suffix_path`.
3. `src/operators/irrepkey.jl` — `ITOKey{I} = (op, bond, vertex)`; `bondcharges`, `vertexlabels`,
   `caterpillar_trees`, `_tree_from_bonds`.
4. `src/operators/irrepmpo.jl` — the output contract, `mpo_terms` (round-trip oracle), `irrep_mpo_tensors`
   (symmetric-tensor assembly), `_bond_space`, `_deg_indices`.
5. `src/datastructures/bipartite.jl` (`min_vertex_cover_bipartite`) and
   `src/datastructures/connectedcomponents.jl` (`bipartite_connected_components`) — reuse these.
6. `src/algorithms.jl` (`BipartiteAlgorithm`, `SVDBondAlgorithm`), `CLAUDE.md`, and `test/test_irrep_*.jl`.

## The non-abelian mapping (the key idea)
ITensor's right vertex = a whole term plus its **additive QN flux**. OpSum's non-abelian analogue is a
term whose per-site alphabet symbol is `ITOKey = (op, bond, vertex)`, where `bond` is the **running
fusion charge out of the site** (the non-abelian generalization of QN flux — a *fusion outcome*, not a
sum) and `vertex` is the fusion multiplicity label.

- `_op_at_ito(tt, t, s)` is exactly ITensor's `get_onsite_op` (it reconstructs the pass-through key
  carrying the running bond charge at idle sites).
- Two terms are "equal from site n on" iff their `ITOKey` suffixes match on **op AND bond AND vertex**
  (`_suffix_path` equality) — this already encodes the fusion channel, so it is the correct non-abelian
  form of ITensor's `are_equal(…, n)`.
- **The graph sweep stays scalar.** Reduced coefficients are `ComplexF64`; min-vertex-cover / SVD
  operate on scalar matrices. Non-abelian structure enters *only* (a) in what makes a bond state
  distinct (the augmented `ITOKey` → `bondsectors` + vertex labels) and (b) at tensor assembly
  (fusion couplers in `irrep_mpo_tensors`). **Do NOT introduce quantum dimensions or F-symbols into the
  sweep.**
- Connected components are pure in the outgoing bond charge (`ITOKey.bond`) — the current code asserts
  this. That purity is what makes per-component decomposition equal to per-sector decomposition, so the
  SVD/QR backend can decompose each component's plain scalar matrix (no graded `TensorMap` inside the
  sweep).

## What to build
1. **Persistent graph type** over `ITOTermTable` (e.g. `ITOGraph{I}`): right vertices = term indices
   (persist across the whole sweep; only their count shrinks via suffix-merge); left vertices = a
   `LeftVertex` analogue `(link::Int, key::ITOKey{I})` (leave a slot for a future fermion/JW flag but do
   not implement it). Adjacency = per-left-vertex lists of `(right_vertex_id, scalar_weight)`.
2. **Seeding** `ITOGraph(tt)`: reverse + sort terms so equal `ITOKey`-suffixes are contiguous (mirror
   `MPOGraph(os)`), merge duplicate terms' coefficients, then bucket left vertices by the site-1 key.
   (`ITOTermTable` already merges coincident active content, but you still need the reverse+sort for the
   incremental suffix-merge to be correct.)
3. **`at_site!`-style step**, five phases exactly per doc §6: (1) suffix-merge right vertices equal from
   n+1 on; (2) connected components via `bipartite_connected_components`; (3) per-component backend; (4)
   assemble the bond (collect `secW` bond charges + per-index vertex labels + offsets); (5) build the
   next graph reusing the same right vertices, new left vertices tagged with the outgoing bond index as
   `link`. Follow the **coefficient-flow rule** in §6 (covered-left forwards the scalar and emits the
   bare letter; covered-right resets weight to 1 and folds `key.op * coeff`) — it matches OpSum's
   current U/V handling in `_irrep_bipartite`.
4. **Both backends** behind the existing selectors: `BipartiteAlgorithm` → per-component
   `min_vertex_cover_bipartite`; `SVDBondAlgorithm` → per-component scalar SVD with the algorithm's
   `trunc`, keeping the left singular vectors as the compressed basis (mirror `_irrep_svd` and ITensor's
   QR path in doc §6 "The QR backend").

## Non-abelian requirements (must hold)
- Preserve and propagate `ITOKey.bond` and `ITOKey.vertex` end to end; `bondsectors[i]` lists the
  outgoing bond charge per bond index, and per-index vertex labels must be recoverable for assembly.
- Assert component bond-charge purity (as the current code does).
- **Reuse `irrep_mpo_tensors` for assembly and `mpo_terms` for the faithfulness check** — produce the
  same `(Ws::Vector{SparseMatrixDOK{LocalOp{ComplexF64,IrrepOperator{I}}}}, bondsectors::Vector{Vector{I}})`
  contract so downstream is unchanged. If the persistent graph naturally carries more (e.g. vertex
  labels), thread it through without breaking that contract, or extend the contract minimally and update
  both consumers.
- Keep arity support at parity with current (K ∈ {0,1,2}). GenericFusion multi-channel and fermionic
  (graded) sectors are **out of scope** unless trivially free — note them as follow-ups.

## Reuse, don't reinvent
`min_vertex_cover_bipartite`, `bipartite_connected_components`, `_op_at_ito`, `_suffix_path`,
`caterpillar_trees` / `bondcharges` / `vertexlabels` / `_tree_from_bonds`, `_bond_space` /
`_deg_indices`, `SparseMatrixDOK`, `increaseindex!`.

## Constraints
- Build alongside; do NOT delete `_irrep_bipartite` / `_irrep_svd` until the new path passes all tests
  and matches them.
- Format every touched file with Runic (see `CLAUDE.md`).
- Tests: add `test/test_irrep_*.jl` following the existing standalone / `ParallelTestRunner` pattern. Use
  `instantiate(ts::TermSum, sites)` as the dense correctness oracle, and **differential-test** the new
  sweep against the existing `_irrep_bipartite` / `_irrep_svd` output (compare via `mpo_terms` round-trip
  and reconstructed dense operators, not raw matrices — the per-sector basis choice may differ). Cover
  the non-abelian sectors already in the suite (`U1Irrep`, `SU2Irrep`).

## Milestones (do in order; verify each before moving on)
1. Persistent graph + seeding + VC `at_site!` reproducing `_irrep_bipartite`'s per-bond dims and passing
   `test/test_irrep_mpo.jl`.
2. Incremental suffix-merge (the ITensor advantage over re-materializing `_suffix_path` each bond);
   benchmark vs the old path on a large Hamiltonian.
3. SVD backend at parity with `_irrep_svd`.
4. Thread vertex labels cleanly; confirm `irrep_mpo_tensors` is unchanged and the `instantiate` oracle
   matches for K=2 non-abelian terms.

## Verification
- Whole suite: `julia --project -e 'using Pkg; Pkg.test()'`
- Targeted: `julia --project test/test_irrep_mpo.jl` (also the alphabet/termtable files)
- Formatting: `julia --project -e 'using Runic; Runic.format_file("src/<file>.jl"; check=true)'`

## Watch-outs (from the research doc)
- The **running-bond-first** fusion coupler ordering in `irrep_mpo_tensors` — charge-first flips signs
  at antisymmetric vertices for K≥3; do not reorder it.
- The pass-through symbol is NOT `one(A)`; it carries the running bond charge (`_op_at_ito`).
- The suffix-merge precondition — equal suffixes must be *contiguous* — is what makes the incremental
  merge correct; get the reversed-`ITOKey` ordering right or the merge silently under/over-collapses.

## Deliverable
Working code behind both algorithm selectors, the new/updated tests, and a short design note (docstring
or a `research/` update) describing the final data structures and any contract change.
