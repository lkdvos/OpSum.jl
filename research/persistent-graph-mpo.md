# Persistent-graph MPO construction — design note

*Outcome of the port described in `research/port-handoff.md`: OpSum's symmetry-reduced MPO
construction re-implemented on ITensorMPOConstruction.jl's persistent bipartite-graph + `at_site!`
sweep architecture (see `research/itensor-mpograph-construction.md`), generalized to the non-abelian
(TensorKit `Sector`) ITO machinery.*

New file: `src/operators/irrepgraph.jl` (included between `irreptermtable.jl` and `irrepmpo.jl`).
The transient-frontier sweeps `_irrep_bipartite` / `_irrep_svd` are **kept unchanged** as the parity
oracles and as the pinned SVD backend (see §4).

## 1. Data structures

```julia
struct LeftVertex{I<:Sector}
    link::Int          # incoming bond index (row into the previous bond's basis)
    key::ITOKey{I}     # on-site ITO key (op, running bond charge, vertex) applied at this site
end                    # ITensor's fermion/JW-string slot is intentionally omitted (no fermions here)

mutable struct ITOGraph{I<:Sector}
    tt::ITOTermTable{I}; N::Int
    # seeding-fixed incremental-merge machinery:
    sortpos::Vector{Int}   # right vertices in reversed-suffix sort order (sorted position -> term id)
    lcp::Vector{Int}       # longest common prefix of consecutive reversed suffix paths
    # persistent right-vertex state (shrinks via suffix-merge):
    rrepr::Vector{Int}     # right vertex -> representative term id
    rhi::Vector{Int}       # right vertex -> max sorted position in its merged class
    # current bipartite graph (bond i-1 -> i), rebuilt each site:
    lefts::Vector{LeftVertex{I}}
    radj::Vector{Vector{Int}}; wadj::Vector{Vector{ComplexF64}}   # adjacency: (right id, scalar weight)
    nlinks::Int            # incoming bond dimension
end
```

The non-abelian mapping (handoff §"the key idea"): a **right vertex is a term** (identified by a
representative term id; they persist across the whole sweep and only merge), and a **left vertex** is
`(incoming link, on-site ITOKey)`. `ITOKey.bond` — the running fusion charge *out of* the site — is
the non-abelian analogue of ITensor's additive QN flux (a fusion *outcome*, not a sum). The graph
sweep itself stays **scalar**: reduced coefficients are `ComplexF64` and the min-vertex-cover / SVD
operate on plain matrices. Non-abelian structure enters only (a) in what makes a bond state distinct
(the augmented `ITOKey` → `bondsectors`) and (b) at tensor assembly (`irrep_mpo_tensors`, unchanged).

## 2. The sweep (`_at_site!`, five phases)

Exactly ITensor's `at_site!` (doc §6), sharing phases 1/2/5 between both backends:

1. **Suffix-merge** (`_suffix_merge!`) — merge right vertices "equal from site `i+1` on". This is the
   incremental win over the transient sweep, which re-materialised every strand's suffix
   (`_suffix_path`) at every bond. Because right vertices stay in the seeding **reversed-suffix sort
   order** and only ever merge, two currently-adjacent right vertices are equal from `i+1` on iff the
   single boundary `lcp[rhi[r]] ≥ N - i` — an O(1) test per boundary (their interiors already satisfy
   the coarser previous threshold, and the threshold only decreases). Total merge work is O(N·M).
2. **Connected components** (`bipartite_connected_components`) — reused verbatim.
3. **Per-component backend** — VC (`_vc_component`) or SVD; see §3/§4.
4. **Assemble the bond** — concatenate component ranks (offsets), collecting `secW` charges and the
   forwarded edges.
5. **Build the next graph** (`_build_next_graph!`) — reuse the same (persistent) right vertices; tag
   fresh left vertices with the outgoing bond index `j` as their `link`, bucketed by `op@(i+1)`.

**Coefficient flow** matches the transient sweep's covered-U / covered-V rule (handoff §3, doc §6):
covered-left forwards its edge weights unchanged and emits the bare letter; covered-right resets the
forwarded weight to 1 and folds `key.op × weight` into the block for every uncovered incident left.
Component bond-charge purity is `@assert`ed, exactly as `_irrep_bipartite` does.

## 3. VC backend — `_irrep_graph_bipartite` (default `BipartiteAlgorithm`)

Per component, `min_vertex_cover_bipartite` chooses the bond basis (covered-left indices first, then
covered-right). This is the wired default: `irrep_mpo(H, sites, BipartiteAlgorithm())` now routes
here.

**Parity:** produces the *same per-sector bond dimensions* and the *same represented operator* as
`_irrep_bipartite` on every Hamiltonian in the suite (`U1Irrep`, `SU2Irrep`, `Trivial`; K ∈ {0,1,2,3};
decoupled multi-component bonds; chains up to N=8). The exact per-sector basis choice / bond-index
ordering may differ (multiple minimum vertex covers of equal size exist), so parity is asserted via
`mpo_terms` round-trip and reconstructed operators, never raw matrices. The whole existing suite
(578 tests) passes with the selector flipped, including `irrep_mpo_tensors` assembly and the
`instantiate` dense oracle for K=2/K=3 non-abelian terms — i.e. the output contract is unchanged.

## 4. SVD backend — `_irrep_graph_svd` (ITensor QR-backend port; not the default)

Implemented and lossless-verified, but **`SVDBondAlgorithm` deliberately still routes to
`_irrep_svd`**. Phases 1/2/5 are shared with the VC step; only the basis choice differs: the whole
bond's scalar coefficient matrix is assembled as a **charge-graded** `TensorMap C : Ppre ← Psuf`
(block-diagonal in the bond charge, so `svd_trunc` does the per-sector SVD *and* the global
across-sector truncation at once), the left singular vectors `U` become the compressed bond basis
(block entry `key.op × U[u,m]`), and `R = S·Vᴴ` forwards the coefficient onto the next bond (folded
into the block at the last site).

**Why not wired:** lossless, it is at parity with `_irrep_svd` (same operator — verified by
contracting the assembled tensors against the dense oracle). Under **truncation the two diverge by
design**: `_irrep_graph_svd` is a *sequential* left-to-right sweep (each bond compressed in the basis
left by the previous bond, à la ITensor's QR sweep), whereas `_irrep_svd` compresses every bond
*independently* on the raw prefix/suffix classes. The existing truncation test pins the
per-bond-independent semantics (e.g. `truncrank(1)` keeps one index *per bond*; the sequential sweep
instead starves downstream bonds after an aggressive early truncation). Rather than change that
pinned behaviour, the sequential variant is kept available and documented. **Follow-up:** expose it
behind a selector once the two truncation semantics are reconciled (or offered as distinct options).

## 5. Contract & scope

- Output contract **unchanged**: both graph functions return
  `(Ws::Vector{SparseMatrixDOK{LocalOp{ComplexF64,IrrepOperator{I}}}}, bondsectors::Vector{Vector{I}})`,
  consumed by `mpo_terms` and `irrep_mpo_tensors` as-is. `ITOKey.vertex` is threaded through
  `LeftVertex.key`, but is `1` throughout the multiplicity-free K ≤ 2 scope, so `bondsectors`
  (charges only) needs no extension.
- Reused verbatim: `min_vertex_cover_bipartite`, `bipartite_connected_components`, `_op_at_ito`,
  `bondcharges`/`vertexlabels`/`caterpillar_trees`/`_tree_from_bonds`, `_bond_space`/`_deg_indices`,
  `SparseMatrixDOK`, `increaseindex!`.
- **Follow-ups** (out of scope, as in the handoff): fermionic/JW strings (the omitted `LeftVertex`
  slot), `GenericFusion` multi-channel (vertex > 1), and wiring the sequential SVD backend (§4).

## 6. Tests

`test/test_irrep_graph.jl` (standalone, discovered by `ParallelTestRunner`):
- `graph VC ≡ transient-frontier bipartite` — identical per-sector bond dims + identical operator +
  lossless round-trip across all reference Hamiltonians.
- `public BipartiteAlgorithm selector uses the graph path`.
- `graph SVD (lossless) reconstructs the operator` — via assembled-tensor contraction vs the dense
  oracle (N ≤ 3, dim-1-total-charge cases).
- `incremental suffix-merge on a longer chain` (N=8) and `right vertices persist while their count
  shrinks` (monotone non-increasing right-vertex count, ending at 1).
