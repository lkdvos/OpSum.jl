# The types (meeting overview, 2026-10-08)

State of `main` at `5f246be`.
See [overview-pipeline.md](overview-pipeline.md) for how they flow into each other.

The types fall into five groups.
Only the first three are things a user normally names; the last two are internal but show up in docstrings, design notes and the #38 discussion.

```
 on-site                  multi-site, latticeless          lattice-bound          output
 IrrepOperator  ─┐        Term ── ITOKey                   OperatorSum            FiniteMPO
 SiteOperator   ─┼──►     Terms  (the compressible op) ──► ├ lattice  ──────────► InfiniteMPO
 LocalOperator  ─┘        ExpSum ── ExpKey                 ├ terms                 └► tensors
                                                           └ channels
```

---

## 1. On-site operators

| type | what it is | fields | made by |
|---|---|---|---|
| `IrrepOperator{I} <: OperatorBasis` | one alphabet letter: an ITO of charge `c`, the `n`-th reduced matrix element of that charge on the site space (TensorKit block order). `n = 0` with `c = unit(I)` is the pass-through identity sentinel. | `c::I`, `n::Int` | `instances(IrrepOperator, V)`; model code should never write `IrrepOperator(c, n)` by hand |
| `SiteOperator{I}` | an unplaced operator on one site: a sparse combination of letters, stored as two parallel vectors (linear scan beats hashing for 1–3 letters) | `letters::Vector{IrrepOperator{I}}`, `coeffs::Vector{ComplexF64}` | `spin`, `spin_ops`, `fermion_ops`, `matrixunit`, `scalarop`, `project(O, V)`, arithmetic |
| `LocalOperator{I}` | an unplaced operator on `K` consecutive slots: a `Terms` bag on sites `1:K`, every term spanning all slots (an idle slot holds the pass-through letter) | `terms::Terms{I}`, `nsites::Int` | `project(h)`, unplaced `couple`/`dot`, `LocalOperator(::SiteOperator)` |
| `OperatorBasis` (abstract) | supertype of alphabets; `IrrepOperator` is the only subtype | — | — |

* A letter has no space attached: it is interpreted relative to whatever space the context supplies (`instantiate(op, V)`).
  That is what lets the term algebra stay latticeless, and also what makes the basis implicit.
* `SiteOperator` is also the **entry type of the reduced MPO's bond matrices**, which is why it is kept separate from `LocalOperator`.
* The `UnplacedOperator{I}` union (`SiteOperator` or `LocalOperator`) is what unplaced `couple`/`dot` accept.

## 2. Terms (multi-site, latticeless)

| type | what it is | fields |
|---|---|---|
| `ITOKey{I}` | one factor of a term: the letter, the running bond charge *after* this site, and the fusion-vertex label (always 1, since only multiplicity-free fusion is supported) | `op::IrrepOperator{I}`, `bond::I`, `vertex::Int` |
| `Term{I}` | one term: ascending unique sites, one key each, one coefficient. The caterpillar fusion tree is derived from the keys (`tree(t)`), not stored. `==`/`hash` ignore the coefficient, so coincident terms can be summed. Never mutated. | `sites::Vector{Int}`, `keys::Vector{ITOKey{I}}`, `coeff::ComplexF64` |
| `Terms{I}` | **the compressible operator**: an unnormalised bag of terms with no lattice. What `A[i]`, `couple`, `dot` and `project(h, sites)` return and what arithmetic combines. Everything that observes it (`length`, iteration, `≈`, `show`) goes through `canonicalize!` first. | `terms::Vector{Term{I}}` |
| `ExpKey{I}` | one exponentially decaying interaction: a representative `Term`, the exit site whose preceding gap is stretched, the decay `λ`, and the on-site string operator | `term`, `exitsite`, `decay`, `string` |
| `ExpSum{I}` | a sum of decaying channels, the geometric counterpart of `Terms` | `channels::Dictionary{ExpKey{I}, ComplexF64}` |

* `Terms + ExpSum` deliberately throws: the two only meet inside an `OperatorSum`.
* `arity(t)` is the number of active sites; `total(t)` the charge it couples to; `ops(t)` its letters.

## 3. Lattices and the lattice-bound sum

| type | what it is | fields |
|---|---|---|
| `AbstractLattice` | supertype of the two chains | — |
| `FiniteChain{S}` | an open chain, one space per site; `getindex` does not wrap and an out-of-range site is an error | `spaces::Vector{S}` |
| `InfiniteChain{S}` | a repeating unit cell of `L` sites; `getindex` wraps, so a term may reach past the cell | `spaces::Vector{S}` |
| `OperatorSum{I, L}` | the one value every MPO entry point takes: a lattice, a `Terms` bag and an `ExpSum`. Terms are validated against the spaces as they enter. | `lattice::L`, `terms::Terms{I}`, `channels::ExpSum{I}` |

* The two chains differ in exactly two observable ways: whether `getindex` wraps, and whether a site outside `1:length` is an error.
* `OperatorSum` is where `adjoint`, `instantiate` and `chain_terms` live, because they need the spaces.

## 4. Outputs

| type | what it is | fields |
|---|---|---|
| `FiniteMPO{I}` | reduced MPO on an open chain; destructures as `Ws, bondsectors` | `Ws::Vector{SparseMatrixCSC{SiteOperator{I}, Int}}`, `bondsectors::Vector{Vector{I}}` |
| `InfiniteMPO{I}` | reduced MPO for one unit cell, plus the indices of the two identity channels per bond; destructures the same way | `Ws`, `bondsectors`, `start::Vector{Int}`, `done::Vector{Int}` |

* Each bond index carries one charge (a one-dimensional level); `length(bondsectors[i])` is the reduced bond dimension and `sum(dim, bondsectors[i])` the dense one.
* The emitted tensors are plain TensorKit / BlockTensorKit types: `irrep_mpo_tensors` gives `TensorMap`s, `jordan_mpo_tensors` gives `SparseBlockTensorMap`s with `SumSpace` virtual legs (one `Vect[I](c => 1)` per index).

## 5. Algorithm selectors and internals

| type | role |
|---|---|
| `BipartiteAlgorithm` | user-facing selector, the default; names `VertexCover` |
| `SVDBondAlgorithm{S}` | user-facing selector; wraps an SVD `BondStrategy` and its truncation |
| `BondStrategy` (abstract) | the plug point of the sweep: how one bond's basis is chosen |
| `VertexCover` | minimum vertex cover per connected component |
| `IndependentSVD{T}`, `SequentialSVD{T}` | per-bond SVD, independent or sequential; `trunc` is a MatrixAlgebraKit truncation or `nothing` |
| `ITOTermTable{I}` (internal) | canonical terms as `K × M` matrices (`sites`, `keys`) plus `coeffs` and `nvertices`; the only input of the sweep |
| `ITOGraph{I}` (internal, mutable) | the persistent bipartite graph handed from one site to the next: suffix-class machinery, right-vertex cursors, the current left vertices and adjacency, lazy-insertion state, start/finish channel bookkeeping and per-site scratch |
| `LeftVertex{I}` (internal) | a prefix: incoming bond index plus the on-site `ITOKey` |
| `ChannelState`, `ExpChannel` (internal) | the lowered automaton of a decaying channel |

---

## Observations for the discussion

* **The type count a user meets is small** (`SiteOperator`, `LocalOperator`, `Terms`, `OperatorSum`, a chain, an MPO), but `IrrepOperator` is exported while the docs say never to construct one.
* **Everything numeric is `ComplexF64`**, including real models; the qchem comparison has to convert back with `realify`.
* **One charge per bond index** is baked into both outputs; `fused_jordan` shows MPSKit accepts multi-sector levels, which the re-blocking idea needs.
* **The letter is the unit of identity throughout** (`ITOKey`, `Term` equality, prefix classes, cover vertices).
  That is why changing what a letter *means* (operator-bases.md) touches so little code: the sweep only ever compares labels.
