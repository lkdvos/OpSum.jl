# MPOGraph construction in ITensorMPOConstruction.jl

*A detailed walk-through of how [ITensorMPOConstruction.jl](https://github.com/ITensor/ITensorMPOConstruction.jl)
(`main` branch) builds an exact, minimal-bond-dimension MPO — with particular attention to the
**site-to-site bookkeeping** — followed by a mapping onto OpSum.jl's existing `_irrep_bipartite`
frontier sweep so the ideas can be reused here.*

This document is a reading aid, not a spec. ITensor references are given as `file:function` (upstream
line numbers drift); OpSum references are `path:line` against the current checkout. Short excerpts are
quoted verbatim from the sources fetched while writing this; nothing here invents an API.

---

## 0. Source-file map

| File | Role |
|---|---|
| `OpIDSum.jl` | Compact fixed-width term storage (`OpID`, `OpInfo`, `OpCacheVec`, `OpIDSum`); ingest of an ITensor `OpSum`. |
| `ops.jl` | `LeftVertex`; the **right-vertex helpers** (`get_onsite_op`, `are_equal`/`terms_eq_from`, `flux`, `is_fermionic`) — the "suffix automaton". |
| `BipartiteGraph.jl` | Generic bipartite graph (`BipartiteGraph`, adjacency helpers, `combine_duplicate_adjacent_right_vertices!`). |
| `connected-components.jl` | `BipartiteGraphConnectedComponents`, `compute_connected_components` (union-find), and `get_cc_matrix` (component → sparse coefficient matrix `W` for the QR path). |
| `large-graph-mpo-common.jl` | **The core.** `MPOGraph`, graph seeding `MPOGraph(os)`, and the per-site step `at_site!`. |
| `large-graph-mpo-vertex-cover.jl` | `process_vertex_cover!` — the `VC` backend. |
| `large-graph-mpo-qr.jl` | `process_qr`, `sparse_qr`, `for_non_zeros_batch`, `process_single_left_vertex_cc!` — the `QR` backend (§6). |
| `MPOConstruction.jl` | Driver: `MPO_new`, `resume_MPO_construction!`, `instantiate_MPO`. |
| `minimum-vertex-cover.jl` | König minimum vertex cover used by `VC`. |

---

## 1. Mental model

An MPO is a **weighted finite-state automaton** read left to right along the chain. Each virtual
(link/bond) index is one automaton state; the local tensor `W_n` is the transition table at site `n`:
entry `W_n[a, s', s, b]` is the amplitude to go from incoming state `a` to outgoing state `b` while
emitting the local operator taking physical index `s → s'`.

Building the MPO therefore means: **at each bond, choose a minimal set of states (a basis for the
link index) that can still represent every term exactly.** ITensor does this by, at each site,
partitioning the operator terms into a *prefix* part (what has happened at and to the left of the
current site) and a *suffix* part (what still must happen strictly to the right), forming a bipartite
graph prefix↔suffix, and taking a **minimum vertex cover** of that graph as the new bond basis. A
vertex cover is exactly a set of "states" that touches every term, and a *minimum* cover is the
smallest such set — hence minimal bond dimension (for a fixed sparsity pattern; see §7).

The distinguishing architectural choice — and the thing this document is really about — is that
ITensor keeps an **explicit, persistent bipartite graph object** (`MPOGraph`) that is handed from one
site step to the next and mutated in place, rather than re-deriving the prefix/suffix partition from
the flat term list at every bond. Crucially:

- **Right vertices = the original terms**, and they *persist* across the whole sweep (only their
  *count* shrinks as suffixes merge).
- **Left vertices are rebuilt each site**; a left vertex remembers only *(which incoming bond index it
  came from, which on-site operator it applies, fermion flag)*.

OpSum.jl, by contrast, rebuilds a transient `frontier` per bond and keeps no graph object between
sites (see §9). The two are mathematically the same algorithm; the persistent graph is a performance
and clarity device.

Two interchangeable backends share all of the graph machinery and differ only in how each connected
component's rank/basis is computed:

- **`VC`** (vertex cover, after Ren, Li *et al.* 2020): minimal bond dimension *among all MPOs with
  the same sparsity pattern*; fast; maximally block-sparse with `splitblocks=true`.
- **`QR`** (rank-revealing decomposition): globally minimal bond dimension in all cases.

We describe `VC` in detail.

---

## 2. Term storage: `OpIDSum` (`OpIDSum.jl`)

Before any graph exists, the Hamiltonian is compiled into a compact, cache-friendly term list.

### Operators are interned

Every distinct local operator on a site is assigned a small integer id and cached:

```julia
struct OpInfo
  name::String
  matrix::Matrix
  is_fermionic::Bool
  qn_flux::QN
end

OpCacheVec = Vector{Vector{OpInfo}}   # op_cache_vec[site][op_id] -> OpInfo
```

By convention **`op_id == 1` is the identity on every site** (`to_OpCacheVec` enforces it). A single
local operator in a term is then just a pair of integers:

```julia
struct OpID{Ti}
  id::Ti   # index into op_cache_vec[n]
  n::Ti    # site
end
Base.zero(::OpID{Ti}) where {Ti} = OpID{Ti}(0, 0)          # padding sentinel
Base.isless(op1::OpID, op2::OpID) = (op1.n, op1.id) < (op2.n, op2.id)
```

### Terms are fixed-width rows

```julia
mutable struct OpIDSum{N,C,Ti}
  _data::Vector{NTuple{N,OpID{Ti}}}   # backing store
  terms::…ReinterpretArray…           # N × capacity matrix VIEW of _data
  scalars::Vector{C}                  # one coefficient per term (column)
  num_terms::Threads.Atomic{Int}
  op_cache_vec::OpCacheVec
  abs_tol::Float64
  modify!::FunctionWrapper…           # optional per-term coefficient callback
end
```

`N` is the maximum number of non-identity local operators in any term (the operator "weight");
storage is a dense `N × capacity` matrix of `OpID`, unused slots padded with `zero(OpID)`. `_data`
and `terms` alias the same memory (`reinterpret(reshape, …)`), which is why terms can be both
row-addressed (`terms[j, i]`) and whole-term-addressed (`_data[i]`) — the latter matters for sorting
(§5).

`add!` (in `ITensorMPS.add!`) does the ingest bookkeeping for one term:

1. drop identity ops (`op.id == 1`);
2. write the remaining ops into the next free column, zero-padding the rest;
3. `sort_fermion_perm!` sorts the ops **by site, stably**, accumulating a `±1` fermionic sign for each
   pair of fermionic operators swapped past one another;
4. apply the optional `modify!` callback;
5. store `scalar * sign * modification`.

`op_sum_to_opID_sum` turns an ITensor `OpSum` into an `OpIDSum`, interning operators on the fly and
setting `N = max(2, max weight)` (the `max(2, …)` avoids a `reinterpret` degeneracy for weight-1
Hamiltonians).

`check_os_for_errors` then validates the global invariants the graph relies on: cached operators are
linearly independent per site, every term is sorted by site with **at most one operator per site**,
and **all terms carry the same total QN flux and the same fermion parity**.

---

## 3. The bipartite graph type (`BipartiteGraph.jl`)

```julia
struct BipartiteGraph{L,R,C}
  left_vertices::Vector{L}
  right_vertices::Vector{R}
  right_vertex_ids_from_left::Vector{Vector{Int}}   # adjacency: per left vertex, list of right ids
  edge_weights_from_left::Vector{Vector{C}}         # parallel weights
end
```

Edges are stored as **per-left-vertex adjacency lists**; `right_vertex_ids_from_left[lv][k]` and
`edge_weights_from_left[lv][k]` are parallel. Duplicate right-vertex ids in one adjacency list are
allowed. `weighted_edge_iterator(g, lv)` zips the two lists.

The one non-trivial operation lives here too:

```julia
combine_duplicate_adjacent_right_vertices!(g, eq) -> new_positions
```

It assumes duplicate right vertices are already **contiguous** (they are, thanks to the seeding sort,
§5), keeps the first of each equal run under predicate `eq`, drops the rest, and remaps every
right-vertex id in the adjacency lists to the survivor. This is the suffix-merge that shrinks the
right side as the sweep advances (§6, phase 1). It is threaded and touches only ids, not weights.

---

## 4. The MPOGraph specialization and its vertex payloads

```julia
MPOGraph{N,C,Ti} = BipartiteGraph{LeftVertex, NTuple{N,OpID{Ti}}, C}
```

so a **right-vertex payload is a whole term** (the `NTuple{N,OpID}` of its operators), and a
**left-vertex payload** is:

```julia
struct LeftVertex
  link::Int32          # incoming bond index (row into the PREVIOUS bond's basis)
  op_id::Int16         # the on-site operator this left vertex applies at the current site
  needs_JW_string::Bool
end
```

This asymmetry is the whole game: the right side is the immutable list of terms; the left side is a
thin, rebuilt-each-site description "*came in on bond index `link`, applied operator `op_id` here*".

### Right vertices are read *relative to a site* (`ops.jl`)

A right vertex is a fixed term tuple, but its meaning at bond `n` is entirely about its **suffix** —
the operators on sites `≥ n`. Four helpers implement this "suffix automaton":

```julia
# op id acting on site n (identity id 1 if none)
function get_onsite_op(ops::NTuple{N,OpID{Ti}}, n)::Ti where {N,Ti}
  for i in 1:N
    ops[i].n == n && return ops[i].id
  end
  return 1
end

# two terms are "equal from site n on": prefixes (ops with .n < n) are ignored
function are_equal(op1::NTuple{N,OpID}, op2::NTuple{N,OpID}, n)::Bool where {N}
  for i in 1:N
    op1[i].n < n && op2[i].n < n && return true    # both suffixes exhausted -> equal
    op1[i] != op2[i] && return false
  end
  return true
end
terms_eq_from(n) = (a, b) -> are_equal(a, b, n)   # curried form for combine_…!

# QN flux / fermion parity of the SUFFIX (ops with .n >= n)
flux(ops, n, cache)        = sum of cache[op.n][op.id].qn_flux      over op.n >= n
is_fermionic(ops, n, cache) = xor of cache[op.n][op.id].is_fermionic over op.n >= n
```

Two subtleties that make `are_equal` work: terms are stored **descending in site** inside the graph
(reversed at seeding, §5), and both are compared position-by-position — so once *both* tuples reach an
op with `.n < n`, their suffixes are identical and everything remaining is prefix, hence "equal".
`terms_eq_from(n+1)` is the predicate driving `combine_duplicate_adjacent_right_vertices!`.

`flux(rv, n+1, …)` is how each surviving bond index gets its QN label; `is_fermionic` and
`needs_JW_string` are how Jordan–Wigner strings are threaded (OpSum's irrep track has no equivalent —
see §9).

---

## 5. Seeding: `MPOGraph(os::OpIDSum)` (`large-graph-mpo-common.jl`)

Three steps turn the flat term list into the initial graph:

**(a) Reverse each term** so its operators run *descending* in site:

```julia
Threads.@threads for i in 1:length(os)
  for j in N:-1:1
    if os.terms[j, i] != zero(os.terms[j, i])
      reverse!(view(os.terms, 1:j, i)); break
    end
  end
end
```

This is what lets `are_equal(…, n)` treat leading positions as suffix and makes equal-suffix terms
sortable into contiguous runs.

**(b) Sort + merge duplicate terms.** A `CoSorter` sorts `_data` (the term tuples) while carrying
`scalars` along, then a linear pass merges byte-identical terms (summing coefficients) and drops
sub-tolerance ones:

```julia
sort!(CoSorter(os._data, os.scalars); alg = … QuickSort)
# … linear dedup: if _data[i] == _data[i+1], fold scalar into i+1, zero i …
```

After this the terms (right vertices) are sorted, so any future equal-suffix runs are contiguous —
the precondition `combine_duplicate_adjacent_right_vertices!` relies on.

**(c) Build the initial left vertices** bucketed by the operator on site 1, via
`build_next_edges_specialization!` + `add_to_next_graph!`. The intermediate structure is a matrix

```julia
next_edges::Matrix{Tuple{Vector{Int},Vector{C}}}   # [m, op_id] -> (right_ids, weights)
```

where **rows `m` are incoming bond indices** and **columns are the next on-site operator id**. For the
seed there is a single incoming index (`m = 1`, the left boundary), and it is called with
`edge_weights = os.scalars`, so **the seed edge weights are the term coefficients** — the start of the
coefficient-flow chain described in §6. `build_next_edges_specialization!` scans every term, computes
`op_id = get_onsite_op(rv, 1)`, and appends `(rv, weight)` to `next_edges[1, op_id]`.
`add_to_next_graph!` then emits one `LeftVertex(m + offset, op_id, needs_JW)` per non-empty bucket,
with that bucket's `(right_ids, weights)` as its adjacency:

```julia
push!(next_graph.left_vertices, LeftVertex(m + cur_offset, op_id, needs_JW_string))
push!(next_graph.right_vertex_ids_from_left, cur_right_vertex_ids)
push!(next_graph.edge_weights_from_left,     cur_edge_weights)
```

The result `g` has: right vertices = all merged terms; left vertices = one per distinct site-1
operator; edges = which terms apply that operator on site 1.

---

## 6. The site step: `at_site!` — the heart of the bookkeeping

```julia
function at_site!(::Type{ValType}, g::MPOGraph, n, sites, tol, absolute_tol,
                  op_cache_vec, alg; combine_qn_sectors, output_level=0)
  :: Tuple{MPOGraph, Vector{Int}, Vector{BlockSparseMatrix{ValType}}, Index}
```

It consumes the graph `g` describing bond `n-1 → n` and returns

- `next_graph` — the graph for bond `n → n+1`,
- `offset_of_cc` — where each connected component's block of bond indices starts,
- `matrix_of_cc` — the local operator blocks for site `n` (a `BlockSparseMatrix` per component),
- `outgoing_link` — the `Index` for the new bond (with QN sectors, or a plain dimension).

`BlockSparseMatrix{C} = Vector{Dictionary{Int,Matrix{C}}}`. Per the upstream docstring, *"the outer
vector is indexed by component-local `right_link`; each inner `Dictionary` maps `left_link` to the
dense local operator matrix for that block."* So: outer index = the **component-local** outgoing bond
index (`1 … rank_of_cc[cc]`), inner key = incoming link, value = the `site_dim × site_dim`
local-operator block. It is the sparse transition table for site `n`, one per component;
`offset_of_cc[cc]` (phase 4) later shifts the component-local indices into the single global bond.

The step has five phases.

### Phase 1 — suffix-merge the right side

```julia
workspace = combine_duplicate_adjacent_right_vertices!(g, terms_eq_from(n + 1))
```

Any two terms whose suffixes agree from site `n+1` on become **one** right vertex: past site `n` they
are indistinguishable (both will only place identities, or the identical remaining operators). This is
where the right side shrinks — a term that has "finished" by site `n` collapses into the shared
"identity to the right" vertex. This is the incremental analogue of recomputing suffix classes from
scratch each bond.

### Phase 2 — connected components

```julia
ccs   = compute_connected_components(g, workspace)
nccs  = num_connected_components(ccs)
```

`compute_connected_components` (`connected-components.jl`) is a union-find over left vertices joined
whenever they share a right vertex. Its result:

```julia
struct BipartiteGraphConnectedComponents
  lvs_of_component::Vector{Vector{Int}}          # left-vertex ids per component
  position_of_rvs_in_component::Vector{Int}      # right id -> local slot within its component
  rv_size_of_component::Vector{Int}
end
```

Because the automaton is block-diagonal in QN flux, **each component is automatically a single QN
sector**, and components are independent — they are processed in parallel (`Threads.@threads for cc`)
and their ranks simply concatenate into the bond. `qi_of_cc[cc]` records the component's QN, read from
a single representative: `flux(right_vertex, n+1)` where the right vertex is the first one reachable
from the component's first left vertex (`g.right_vertex_ids_from_left[ ccs.lvs_of_component[cc][1] ][1]`).
Sector-purity of the component is what makes that single sample well-defined.

### Phase 3 — cover each component (`process_vertex_cover!`)

For each component the `VC` backend computes a **minimum vertex cover**:

```julia
left_cover, right_cover = minimum_vertex_cover(g, ccs, cc)
rank = length(left_cover) + length(right_cover)
rank_of_cc[cc] = rank
```

`minimum_vertex_cover` (`minimum-vertex-cover.jl`) is Hopcroft–Karp maximum matching + König, run on
the single component; it returns **component-local, ascending-sorted** index lists (local ids into
`ccs.lvs_of_component[cc]` on the left; into the component's right-vertex slots
`position_of_rvs_in_component` on the right — *not* global graph ids). `rank` is the component's
contribution to the new bond dimension. The cover partitions into two kinds of new bond index, and this
partition **is** the site-to-site handoff:

- **Covered *left* vertices** → this bond index *originates* a local operator at site `n`. Its block
  is filled from the left vertex's own operator:

  ```julia
  local_op = op_cache[lv.op_id].matrix
  add_to_local_matrix!(matrix_element, one(ValType), local_op, lv.needs_JW_string)
  set!(matrix[m], lv.link, matrix_element)     # keyed by incoming link
  ```

  Every **uncovered** left vertex must then be represented against some covered *right* vertex it is
  adjacent to; its weighted operator is added into that right vertex's column instead:

  ```julia
  m = right_cover_m[position_of_rvs_in_component[rv_id]]
  add_to_local_matrix!(matrix_element, weight, local_op, lv.needs_JW_string)
  ```

- **Covered *right* vertices** → this bond index *carries a suffix forward*. It gets a single
  outgoing edge of unit weight to its representative term.

  Note the invariant that makes the uncovered-left step well-defined: if a left vertex is *not*
  covered, every one of its edges must be covered by its right endpoint (vertex-cover property), so
  every neighbour right vertex of an uncovered left is guaranteed to be a covered right — hence
  `right_cover_m[position_of_rvs_in_component[rv_id]]` always resolves.

**Bond-index layout within a component.** The `rank` new bond indices are laid out in a fixed order:
covered-left vertices take component-local indices `1 … length(left_cover)`, covered-right vertices
take `length(left_cover)+1 … rank` (`right_cover_m[local_rv] = length(left_cover) + m`). `matrix_of_cc[cc]`
and the rows of `next_edges` are both indexed by this same `1 … rank`, so bond index, local block, and
outgoing-edge row all line up. `offset_of_cc[cc]` then maps `1 … rank` into the global `outgoing_link`.

**Where the coefficient lives** (the subtlest bit of the bookkeeping). A term's scalar rides on its
graph *edge weight* (seeded from `os.scalars`, §5). It is deposited into a local operator block exactly
once, at the site where the term's left vertex is *uncovered and folded onto a covered right vertex*
(`add_to_local_matrix!(matrix_element, weight, …)` — weighted). A **covered-left** block is instead
filled with **unit** weight (`add_to_local_matrix!(matrix_element, one(ValType), …)`) and its edges
push the weight *forward* unchanged; a **covered-right** forward edge resets the weight to `one(C)`.
So a term that keeps getting left-covered keeps forwarding its coefficient until, at the site where it
finally merges into a shared suffix (or at the last site, where every left is uncovered and the single
right vertex is covered), the coefficient lands in that site's block. This mirrors OpSum's strand rule
(§9.2): covered-U forwards the residual coeff and emits a bare letter; covered-V resets to 1 and its
incident U's emit `op × coefficient`.

`add_to_local_matrix!` also applies the Jordan–Wigner sign: when `needs_JW_string`, it negates the
odd-parity columns of the local operator (the `size == 2` and `size == 4` branches).

**Building the next edges.** Still inside `process_vertex_cover!`, the outgoing adjacency for the
*next* site is assembled into a fresh `next_edges[rank, n_ops_next]` matrix, precomputing
`next_op_of_rv_id[rv] = get_onsite_op(rv, n+1)`:

- covered-left rows keep **all** their term edges, bucketed by next on-site op:

  ```julia
  op_id = next_op_of_rv_id[rv_id]
  push!(next_edges[m, op_id][1], rv_id); push!(next_edges[m, op_id][2], weight)
  ```

- covered-right rows get **one** unit edge to their representative term:

  ```julia
  resize!(next_right_vertex_ids, 1); resize!(next_edge_weights, 1)
  next_right_vertex_ids[1] = rv_id;   next_edge_weights[1] = one(C)
  ```

`next_edges_of_cc[cc] = next_edges` stores this per component.

### Phase 4 — assemble the bond index

```julia
for cc in 1:nccs
  qi_of_cc[cc] = first(qi_of_cc[cc]) => rank_of_cc[cc]     # QN => multiplicity
end
cc_order = combine_qn_sectors && has_qns ? merge_qn_sectors(qi_of_cc)… : 1:nccs
outgoing_link = has_qns ? Index(qi_of_cc; tags="Link,l=$n", dir=Out)
                        : Index(cur_offset; tags="Link,l=$n")
```

`offset_of_cc[cc]` records where component `cc`'s `rank_of_cc[cc]` bond indices sit inside the single
`outgoing_link`; `merge_qn_sectors` optionally reorders/coalesces components sharing a QN.

### Phase 5 — build `next_graph` (the handoff)

```julia
next_graph = MPOGraph{N,C,Ti}([], g.right_vertices, [], [])   # SAME right vertices
cur_offset = 0
for cc in cc_order
  offset_of_cc[cc] = cur_offset
  add_to_next_graph!(next_graph, g, op_cache_vec, n, cur_offset, next_edges_of_cc[cc])
  cur_offset += rank_of_cc[cc]
end
```

`next_graph` **reuses `g.right_vertices` verbatim** — the terms persist. Its new left vertices are
created by `add_to_next_graph!` from `next_edges`:

```julia
needs_JW_string = is_fermionic(right_vertex(cur_graph, first_rv_id), n + 2, op_cache_vec)
push!(next_graph.left_vertices, LeftVertex(m + cur_offset, op_id, needs_JW_string))
```

The new left vertex's `link = m + cur_offset` is precisely the **outgoing bond index this partial term
occupies at bond `n`** — i.e. `LeftVertex.link` is the literal pointer that stitches bond `n-1 → n` to
bond `n → n+1`. That is the entire cross-site bookkeeping: everything else (coefficients, operators,
QN) is recomputed locally, but the identity of "which bond index continues which partial term" is
carried in `LeftVertex.link`.

> **In one sentence:** a covered *left* vertex means "a term *starts* its operator here and continues
> right on this bond index"; a covered *right* vertex means "a shared suffix flows through here on this
> bond index"; and `next_graph`'s left vertices, tagged with `link = offset + row`, encode which of
> those bond indices each term will ride into the next site.

### The `QR` backend — the other phase 3 (`large-graph-mpo-qr.jl`)

`process_qr` is a drop-in replacement for `process_vertex_cover!`: **identical signature**, fills the
same `matrix_of_cc` / `rank_of_cc` / `next_edges_of_cc`, and phases 1, 2, 4, 5 of `at_site!` are byte
-for-byte the same. The only difference is *how a component's bond basis is chosen* — a rank-revealing
sparse QR instead of a vertex cover — which yields the **globally minimal** bond dimension (VC is
minimal only among MPOs of the same sparsity pattern). Per component (`Threads.@threads for cc`):

1. **Single-left-vertex fast path.** If `left_size(ccs, cc) == 1`, `process_single_left_vertex_cc!`
   sets `rank = 1`, writes one **unit**-weight block keyed by `lv.link`, and forwards edges via
   `build_next_edges_specialization!` (exactly the seeding path). At the last site it folds the lone
   edge weight, `scaling = only(g.edge_weights_from_left[lv_id])`, into the block. This skips a QR for
   the ubiquitous trivial component.

2. **Assemble the coefficient matrix `W`.** `get_cc_matrix(g, ccs, cc; clear_edges=true)` (defined in
   `connected-components.jl`) builds a `SparseMatrixCSC{C}` of size `num_left × num_right`:

   - **rows index the component's left vertices** (`left_map[i]` → global left id),
   - **columns index its right vertices** (`right_map[j]` → global right id),
   - entry `W[i, j]` = the edge weight coupling them, with **duplicate edges summed** (same semantics as
     `sparse(row, col, val)`).

   `clear_edges=true` releases each left vertex's adjacency as it is consumed (memory). `W` is exactly
   the prefix↔suffix coefficient matrix of the component — the same object VC covers, here decomposed.

3. **Rank-revealing QR.** `sparse_qr(W, tol, absolute_tol)` calls SuiteSparse SPQR (`qr(A; tol)`,
   column-pivoted) and returns `Q, R, prow, pcol, rank`. `tol` is scaled by `SPQR._default_tol(A)`
   unless `absolute_tol`, and the numerical `rank = rank(ret)` **is the new component bond dimension**.
   With the row/column pivots `prow`/`pcol`, `W` factors as `Q * R` up to those permutations (the code
   never forms the equation — it uses `prow` to map `Q` rows back to left vertices and `pcol` to map
   `R` columns back to right vertices, which is all that matters below).

4. **`Q` columns → local operator blocks.** The first `rank` columns of `Q` are the new left-side
   basis. `for_non_zeros_batch(Q, rank)` visits each column `m` (densified from the Householder factors
   `Q.factors` / `Q.τ` by `get_column!`); each nonzero `Q[i, m] = weight` accumulates
   `weight · op(left_vertex(g, left_map[prow[i]]))` into `matrix[m][lv.link]` via `add_to_local_matrix!`.
   So **each new bond index `m` is a linear combination of the incoming left vertices** — unlike VC,
   where a covered bond index is a *single* prefix/suffix vertex. This is why QR tensors are denser but
   the bond can be strictly smaller.

5. **`R` rows → next edges.** `R` (size `rank × num_right`) expresses each suffix in the new basis.
   `for_non_zeros_batch(R, length(right_map))` walks column `j`; `rv_id = right_map[pcol[j]]`,
   `op_id = get_onsite_op(rv, n+1)`, and each nonzero `R[m, j] = weight` pushes `(rv_id, weight)` into
   `next_edges[m, op_id]`. So **the forwarded edge weights are the `R` entries** — the coefficient of
   each suffix in the compressed basis. (Contrast VC: a covered-right vertex forwarded a *unit* weight.)

6. **Last site.** When `n == length(sites)` there is one trivial suffix, `R` is effectively a scalar
   `scaling = only(R)`; every block is multiplied by it and `next_edges` is skipped.

**VC vs QR in one line.** VC picks a *subset* of vertices as the basis (each bond index = one prefix or
one suffix, unit coupling → sparser tensor, minimal-for-fixed-sparsity); QR picks an *orthonormal linear
combination* (each bond index = a `Q`-column mixture, `R`-weighted forwarding → denser tensor, globally
minimal rank). The coefficient bookkeeping also differs: VC deposits a term's scalar at the
uncovered→covered fold, whereas QR spreads it across the `Q` (block) and `R` (forwarded-edge) factors
whose product reconstructs `W`. Either way the cross-site machinery (`LeftVertex.link`, `offset_of_cc`,
`next_graph`) in `at_site!` is untouched, and both MPOs are exact.

---

## 7. Driver and tensor assembly (`MPOConstruction.jl`)

```julia
function MPO_new(ValType, os::OpIDSum, sites; …)
  prepare_opID_sum!(os, …); check_os_for_errors && check_os_for_errors(os)
  g = MPOGraph(os)                                       # §5

  llinks = Vector{Index}(undef, length(sites) + 1)
  llinks[1] = hasqns(sites) ? Index(QN()=>1; tags="Link,l=0", dir=Out) : Index(1; tags="Link,l=0")

  resume_MPO_construction!(1, offsets, block_sparse_matrices, sites, llinks, g, os.op_cache_vec; …)
  return instantiate_MPO(offsets, block_sparse_matrices, sites, llinks; splitblocks, checkflux)
end
```

The sweep is a plain loop; note that `at_site!` writes `llinks[n+1]` — the outgoing link becomes the
next site's incoming link, so `llinks` is the shared spine:

```julia
for n in n_init:length(sites)
  g, offsets[n], block_sparse_matrices[n], llinks[n + 1] =
      at_site!(ValType, g, n, sites, tol, absolute_tol, op_cache_vec, alg; combine_qn_sectors, output_level)
  call_back(n, …)
end
```

`instantiate_MPO` then materializes each tensor from its block-sparse matrix and the two adjacent
links, and contracts away the dummy boundary links:

```julia
H[n] = to_ITensor(offsets[n], block_sparse_matrices[n], llinks[n], llinks[n+1], sites[n]; splitblocks)
…
L = ITensor(llinks[1]);      L[end] = 1.0      # left boundary row vector
R = ITensor(dag(llinks[end])); R[1] = 1.0      # right boundary column vector
H[1] *= L;  H[length(sites)] *= R
```

`resume_MPO_construction!` takes an `n_init` and a `call_back`, so the sweep is resumable/streamable —
useful for very large graphs, and the reason the state (`offsets`, `block_sparse_matrices`, `llinks`,
`g`) is passed as plain vectors rather than captured in a closure.

**Backend choice.** `alg="VC"` runs `process_vertex_cover!` (§6, phase 3); `alg="QR"` runs
`process_qr` (§6, "The `QR` backend"). The `tol` / `absolute_tol` arguments threaded through `at_site!`
are **used only by the QR path** (the SPQR numerical-rank cutoff); `VC` ignores them and is exact by
construction.

---

## 8. Worked example: 3-site transverse-field Ising chain

Take, on 3 sites,

```
H = J·(Sz₁Sz₂ + Sz₂Sz₃) + h·(Sx₁ + Sx₂ + Sx₃)
```

Per-site op cache: `id 1 = I`, `id 2 = Sz`, `id 3 = Sx`. The expected answer is the textbook TFIM MPO
with bond dimensions **1 – 3 – 3 – 1**; let us watch the graph produce it.

### 8.1 `OpIDSum` (after `add!`, sorted ascending by site)

| term | scalar | ops (`OpID(id,n)`) |
|---|---|---|
| T1 | J | (2,1)(2,2) |
| T2 | J | (2,2)(2,3) |
| T3 | h | (3,1) |
| T4 | h | (3,2) |
| T5 | h | (3,3) |

### 8.2 Seed graph `MPOGraph(os)`

Terms are reversed (descending site) and sorted; then left vertices are bucketed by the op on site 1.
`get_onsite_op(·, 1)`: T3→Sx, T1→Sz, and T2,T4,T5→I (nothing on site 1). So:

```
MPOGraph at site 1: 3 left vertices, 5 right vertices
  Left vertices:                          Right vertices (suffix view):
    L_a: link=1, op=I    -> {T4,T2,T5}      T1: Sz¹ Sz²      T4: Sx²
    L_b: link=1, op=Sz   -> {T1}            T2: Sz² Sz³      T5: Sx³
    L_c: link=1, op=Sx   -> {T3}            T3: Sx¹
```

(all three left vertices carry `link=1`, the single left boundary index.)

### 8.3 `at_site!(n = 1)`

**Phase 1** — `terms_eq_from(2)`: suffixes from site 2 are `{}` (T3), `{Sz²}` (T1), `{Sx²}` (T4),
`{Sz²Sz³}` (T2), `{Sx³}` (T5) — all distinct, no merge.

**Phase 2** — components: `{L_a}↔{T4,T2,T5}`, `{L_b}↔{T1}`, `{L_c}↔{T3}` (no shared right vertices).

**Phase 3** — cover per component. `{L_a}↔{T4,T2,T5}` is a star `K₁,₃`; its minimum cover is the single
**left** center. The other two are single edges (cover either endpoint — the left is chosen). So
`left_cover` in all three, `rank = 1` each, **bond dim after site 1 = 3**. The three bond indices mean:

| bond index | meaning | filled from |
|---|---|---|
| 1 | "not started yet" (identity so far) | L_a (op I) |
| 2 | "placed Sz on site 1, awaiting Sz" | L_b (op Sz) |
| 3 | "finished: Sx¹ done, identity to the right" | L_c (op Sx) |

**Phase 5** — `next_graph`: each covered left keeps its edges bucketed by `get_onsite_op(·, 2)`,
producing site-2 left vertices (link = the bond index above):

```
  (link=1, op=Sx) -> {T4}      (from "not started", place Sx on site 2)
  (link=1, op=Sz) -> {T2}      (from "not started", start Sz on site 2)
  (link=1, op=I ) -> {T5}      (from "not started", still idle)
  (link=2, op=Sz) -> {T1}      (finish the Sz¹Sz² bond)
  (link=3, op=I ) -> {T3}      (finished term, propagate identity)
```

### 8.4 `at_site!(n = 2)`

**Phase 1** — `terms_eq_from(3)`: suffixes from site 3 are `{}` for T3, **T1, T4** (all finished by
site 2) and `{Sz³}` (T2), `{Sx³}` (T5). So **T1, T3, T4 merge into one "finished" right vertex**; T2
and T5 stay. This is the suffix-merge shrinking the right side from 5 to 3.

**Phase 2** — components:

```
  C1 = {(1,Sx),(2,Sz),(3,I)} ↔ {finished}     # three lefts, one shared right
  C2 = {(1,Sz)}              ↔ {T2}
  C3 = {(1,I)}               ↔ {T5}
```

**Phase 3** — cover. C1 is a star centered on the **right** vertex, so its minimum cover is that single
**right** vertex (`right_cover`, rank 1) — the three incoming lefts are all represented against it.
C2, C3 are single edges (rank 1 each). **Bond dim after site 2 = 3**:

| bond index | kind | meaning |
|---|---|---|
| 1 | right-covered | "finished" — everything left of bond 2 is complete |
| 2 | left-covered | "placed Sz on site 2, awaiting Sz on site 3" (T2) |
| 3 | left-covered | "not started" — still idle, term Sx³ lies ahead (T5) |

Note the contrast with site 1: here the shared "finished" channel is captured by covering **one right
vertex** instead of three left vertices — that is the compression that keeps the interior bond at 3
rather than letting it grow.

### 8.5 Site 3 and result

At `n = 3` (the last site) all remaining terms terminate; the outgoing link has dimension 1 (the right
boundary). Final MPO bond dimensions:

```
   1 ── 3 ── 3 ── 1
 (bd0) (site1) (site2) (bd3)
```

exactly the textbook TFIM MPO. The example exercised every mechanism: suffix-merge (T1,T3,T4 → one
vertex), connected components, **left-cover** (site 1) vs **right-cover** (site 2), and the
`LeftVertex.link` handoff between bonds.

---

## 9. Bridge to OpSum.jl

OpSum.jl's `_irrep_bipartite` (`src/operators/irreptermtable.jl:114`) implements the *same* algorithm
with a **transient `frontier`** instead of a persistent graph. The frontier is rebuilt each bond and
discarded; only `sizes`, per-site `dicts`, and `bondsectors` persist
(`src/operators/irreptermtable.jl:122-126`).

```julia
Strand  = Tuple{Int, T}                                     # (representative term, residual coeff)
frontier = [Tuple{Int,T}[(t, tt.coeffs[t]) for t in 1:M]]   # one Vector{Strand} per incoming bond index
```

### 9.1 Correspondence table

| ITensorMPOConstruction (`file:function`) | OpSum.jl (`path:line`) |
|---|---|
| persistent `MPOGraph`; terms = right vertices, kept across sweep | transient `frontier::Vector{Vector{Strand}}`, rebuilt per bond — `irreptermtable.jl:122`, reset at `:240` |
| `LeftVertex{link, op_id, needs_JW}` | the parallel arrays `Uop`/`Uleft`/`Ustrands` — `irreptermtable.jl:130-143` |
| right vertex = whole term `NTuple{N,OpID}` | materialized suffix `_suffix_path(tt,t,i,N)` + representative `Vrepr[iv]` — `irreptermtable.jl:105-107`, `:147-165` |
| `get_onsite_op(ops, n)` | `_op_at_ito(tt, t, s)` (pass-through reconstructs the running **bond charge**) — `irreptermtable.jl:88-102` |
| `are_equal(…, n)` / `terms_eq_from` suffix-merge | grouping strands by `_suffix_path` then `bipartite_connected_components` — `irreptermtable.jl:154-165`, `:185` |
| `flux(ops, n, …)` QN of suffix | `ITOKey.bond` running charge → `secW` → `bondsectors[i]` — `irrepkey.jl:48-52`, `irreptermtable.jl:192-237` |
| `is_fermionic` + JW column flips in `add_to_local_matrix!` | **no equivalent** (irrep track has no fermions) — extension point |
| `minimum_vertex_cover` per component (König) | `min_vertex_cover_bipartite` per component (Hopcroft–Karp + König) — `bipartite.jl:118-163`, called at `irreptermtable.jl:187` |
| components before cover (`compute_connected_components`) | `bipartite_connected_components(adjU, nV)` — `irreptermtable.jl:185` |
| `BlockSparseMatrix` + `to_ITensor` | `SparseMatrixDOK` + `irrep_mpo_tensors` — `irreptermtable.jl:231-243`, `irrepmpo.jl` |
| `offset_of_cc` / `outgoing_link` QN sectors | `secW` → `bondsectors[i]`, `_bond_space` / `_deg_indices` in `irrepmpo.jl` |
| `QR` backend (`large-graph-mpo-qr.jl`) | `SVDBondAlgorithm` / `_irrep_svd` — `irreptermtable.jl:246+` |

### 9.2 The precise mapping of the cover handoff

OpSum's covered-U and covered-V loops (`irreptermtable.jl:198-227`) are exactly ITensor's covered-left
and covered-right cases:

- **Covered U** (`:199-209`) pushes the U vertex's strands forward with their residual coefficients
  and emits the bare letter (`convert(LOp, k.op)`) — a term *starting* its operator, coefficient
  carried forward in the strand. ≙ ITensor **covered left** (`set!(matrix[m], lv.link, …)` with unit
  weight, edges kept).
- **Covered V** (`:211-227`) pushes a single strand `(Vrepr[iv], one(T))` — coefficient reset to 1
  because it is now absorbed to the left — and emits `Uop[iu].op * coefficients[iu,iv]` for every U on
  that suffix. ≙ ITensor **covered right** (unit outgoing edge to the representative term; incoming
  weights folded into the block).

The sector-purity assertion at `irreptermtable.jl:220` is the OpSum statement of ITensor's
"each connected component is one QN sector" fact (§6, phase 2).

### 9.3 Concrete notes for future OpSum work

1. **Incremental suffix-merge.** ITensor's biggest structural advantage is
   `combine_duplicate_adjacent_right_vertices!`: it merges finished suffixes *incrementally* on a
   persistent right-vertex list, whereas OpSum re-materializes the full suffix path for every strand at
   every bond (`_suffix_path`, called in the hot loop at `irreptermtable.jl:154`). Adopting a retained,
   term-indexed right-vertex list (keyed by the reversed term, merged with a `terms_eq_from`-style
   predicate) would remove that repeated O(N) suffix materialization.

2. **`_irrep_svd` is the natural home for a persistent graph.** The SVD path already precomputes global
   per-term transition classes `pre_ids[t,b]` / `suf_ids[t,b]` (`irreptermtable.jl:246+`) — a
   materialized prefix/suffix automaton indexed by term, which is the closest existing analogue to
   ITensor's persistent `MPOGraph`. If OpSum wants ITensor-style incremental bookkeeping, generalize
   that structure and let the bipartite (`VC`) path consume it too, rather than rebuilding `frontier`.

3. **Components-before-cover is already shared** — both run connected components first and cover each
   component independently (identical König construction). No change needed; it is the natural place to
   parallelize (ITensor threads over components in `at_site!`).

4. **`LeftVertex.link` vs the `frontier` index.** OpSum already carries the equivalent pointer as the
   position of each `next_frontier` entry (the `j` in `irreptermtable.jl:201`/`:213`) and records
   `Uleft[iu]` as the incoming index. A persistent-graph refactor should keep an explicit
   `link`-style field rather than relying on positional `frontier` indices, to make the cross-site
   pointer first-class (and easier to debug/serialize, à la `resume_MPO_construction!`).

5. **Fermions / JW strings are the one genuine feature gap.** ITensor threads fermion parity end to
   end: `is_fermionic` on suffixes, `needs_JW_string` on `LeftVertex`, the column sign-flips in
   `add_to_local_matrix!`, and the permutation sign in `sort_fermion_perm!`. OpSum's irrep track has no
   analogue. If fermionic operators are ever needed, this is the machinery to port — and note that in
   the symmetric setting the JW string is subsumed by the sector structure, so the port is not
   mechanical.

---

## 10. One-paragraph summary

ITensorMPOConstruction builds the MPO as a finite-state automaton by maintaining one persistent
bipartite graph whose right vertices are the (merged, reversed) Hamiltonian terms and whose left
vertices are thin `(incoming-link, on-site-op, fermion-flag)` records. Each `at_site!` call
suffix-merges the right vertices for the coming site, splits the graph into QN-pure connected
components, takes a minimum vertex cover of each (the `VC` backend) to choose that bond's basis, writes
the site's local operator blocks, and produces the next graph — reusing the same right vertices and
tagging fresh left vertices with the outgoing bond index (`LeftVertex.link`) that each partial term
will ride into the next site. OpSum.jl already runs the identical algorithm in `_irrep_bipartite`, but
with a transient per-bond `frontier` in place of the persistent graph; the highest-value idea to borrow
is ITensor's incremental suffix-merge on a retained, term-indexed right-vertex list.
