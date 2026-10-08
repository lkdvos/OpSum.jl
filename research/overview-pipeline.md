# The pipeline, end to end (meeting overview, 2026-10-08)

State of `main` at `5f246be`.
Companion notes: [overview-types.md](overview-types.md) for the data types, [overview-interface.md](overview-interface.md) for what a user writes, and [operator-bases.md](operator-bases.md) for the proposed change.

OpSum turns a sum of symmetric operators into a symmetry-reduced MPO.
It does so in six stages, each with one job and one data type handed to the next:

```
 on-site operators      placed terms         lattice-bound sum     flat table         reduced MPO            tensors
 SiteOperator      ──►  Terms           ──►  OperatorSum      ──►  ITOTermTable  ──►  FiniteMPO /       ──►  TensorMap / SparseBlockTensorMap
 LocalOperator          (couple, dot, A[i])  (opsum(lat, …))       (canonicalize!)    InfiniteMPO            (irrep_mpo_tensors / jordan_mpo_tensors)
   1. build               2. place & couple    3. bind & validate    4. normal form     5. compress            6. emit
```

The guiding constraint is that each stage needs as little as possible from the physical problem:

| stage | needs the sector type | needs `N` | needs the physical spaces |
|---|---|---|---|
| 1. build on-site operators | yes | no | the one space it is built on |
| 2. place and couple | yes | no | no |
| 3. bind to a lattice | yes | yes | **yes**: letters are checked here |
| 4. normal form | yes | yes | no |
| 5. compress | yes | yes | no |
| 6. emit tensors | yes | yes | **yes** |

Stages 2, 4 and 5 never see an `ElementarySpace`.
That is a design choice, not an accident: it keeps the term algebra and the sweep independent of the physical spaces, and is the reason the on-site-basis proposal can be confined to stages 1, 3 and 6.

---

## 1. Build on-site operators

The input is a TensorKit space `V` and either a named convention or a dense on-site tensor.
The output is a `SiteOperator`: a short list of ITO *letters* with coefficients.

* A letter is an `IrrepOperator(c, n)`: charge `c`, and `n` indexing the reduced matrix elements of that charge on `V` in TensorKit's block order.
  `instantiate(letter, V)` turns it into a `V ← V ⊗ C_c` tensor with a single nonzero reduced element.
  The letters therefore form a matrix-unit basis of the operators on `V`, orthonormal under TensorKit's inner product.
* Builders: `spin(V)` (SU(2) vector operator, one letter), `spin_ops(V, …)` (U(1)-graded `Sp, Sm, Sz`), `fermion_ops(V, …)` (`c, cd, n`), `matrixunit(V, out, in)`, `scalarop(c, V)` (identity), and the general route `project(O, V)` from any dense symmetric `TensorMap`.
* Multi-site blocks go through `project(h)` on a `K`-site `TensorMap`, giving an unplaced `LocalOperator` on slots `1:K`.

There is no symbolic on-site product.
A product is built as a `TensorMap` and projected back.

## 2. Place and couple

`A[i]` places a `SiteOperator` on site `i`, and `B[i]` or `B[s₁, …, s_K]` places a `LocalOperator`.
Both give a `Terms` bag.
`couple(a, b; to = c)` and `dot(a, b)` combine bags into multi-site terms.

* A `Term` stores its sites in ascending order, one `ITOKey` per site (the letter, the running bond charge after that site, and the vertex label), and one coefficient.
  For the left-nested (caterpillar) coupling used throughout, the running bond charges *are* the fusion tree, so no tree is stored.
* `couple` **distributes**: composite operands produce one term per letter combination whose charges can reach `to`.
  This is the step that splits `c† = L₁ + L₂` into separate letter terms (see operator-bases.md §1).
* Under an abelian symmetry operands may come in any site order.
  The out-of-order leg is inserted and the coefficient picks up the R-symbol, which is the fermionic sign for odd charges.
  Coupling two fermion-odd operands also adds the graded-tensor-product `−1`.
* Non-abelian `couple` must be written in site order and nested to name each intermediate channel.
  `dot` is the exception and accepts either order for any symmetry.

Nothing is deduplicated here; terms are appended as given.

## 3. Bind to a lattice and validate

`opsum(lat, terms...)` creates an `OperatorSum`: the lattice, a `Terms` bag and an `ExpSum` of exponentially decaying channels (`expterm`).
`opsum!` appends in place.

* The lattice is a `FiniteChain` or an `InfiniteChain` (a repeating unit cell).
* On insertion `_checkinsert` validates the sector, the letters against each site's space, and the site range.
  On an `InfiniteChain` it also requires every term to be charge-neutral with at least one site.
* This is the one container that knows the spaces, so the space-dependent operations live here: `H'` (via materialise → adjoint → project, memoised per term shape), `instantiate(H)` (the dense oracle) and `chain_terms(H)`.
* On an `InfiniteChain` the sum is a *generating set*: one representative per translation class, checked by `unitcell_terms` when the MPO is formed.

## 4. Normal form and flat storage

`canonicalize!` sorts the terms, sums coincident ones (equal sites and keys) and drops cancelled ones, in place.
It is the only place the normal form is taken, and it assumes nothing about its input.
`ITOTermTable(ts, N)` then lays the canonical terms out as `K × M` matrices (`sites`, `keys`) plus a coefficient vector.
This table is the only thing the compression reads.
Prefix and suffix classes of the columns are interned once (`_prefix_ids`, `_suffix_ids`) and shared by all backends.

## 5. Compress, bond by bond

The sweep walks the chain once, from left to right, carrying a persistent bipartite graph (`ITOGraph`) between sites.
At each bond:

* **Left vertices** are prefixes: an incoming bond index together with the `ITOKey` applied on this site.
* **Right vertices** are suffix classes, named by an interned `(remaining factors, running bond charge)` signature rather than a materialised path.
  Terms enter lazily at their first active site, so the cost is `Θ(M·K) + Θ(Σ span)`, linear in `N` for finite-range models.
* Edges carry the scalar coefficients.
  The graph is block-diagonal in the bond charge, and each connected component is solved independently.
* A **`BondStrategy`** chooses the bond's basis:
  * `VertexCover` (the default, selected by `BipartiteAlgorithm()`): minimum vertex cover (Hopcroft–Karp + König).
    It is lossless and minimal *for the given sparsity pattern*.
    Covered-left vertices emit their letter and forward their edges; covered-right vertices fold `letter × weight` of every uncovered incident left vertex into one entry.
  * `IndependentSVD` / `SequentialSVD` (selected by `SVDBondAlgorithm(trunc; sweep)`): rank-revealing per bond, optionally truncating.
    `truncrank(k)` means `k` per bond for the independent sweep, and `k` after upstream truncation for the sequential one.
* Infinite chains unroll a window, run the same sweep, and look for the first unit cell that reaches a translation-invariant fixed point.
  This depends on an always-on canonical ordering of each bond.
* Exponentially decaying channels become small weighted automata whose cyclic states are forced into the cover, so `λ` lands on a diagonal pass-through.

The result is a `FiniteMPO` or `InfiniteMPO`: per site a sparse matrix of `SiteOperator` entries (letters times reduced coefficients), and per bond the charge of each index.
`mpo_terms` inverts this at the reduced level; `islossless(H)` is `mpo_terms(irrep_mpo(H)) ≈ chain_terms(H)`.

## 6. Emit tensors

* `irrep_mpo_tensors(mpo, lat)`: one dense symmetric `TensorMap` per site, built with one contraction per distinct letter per site rather than per entry.
* `jordan_mpo_tensors(H[, alg])`: one `BlockTensorKit.SparseBlockTensorMap` per site, with one *level* per bond index ordered `(start, …, finish)`, identities at the two corners, and unit pass-throughs emitted as `BraidingTensor`s so that MPSKit stores them as scalars.
  This is what MPSKit's `JordanMPOTensor` / `FiniteMPOHamiltonian` consume.
  It is defined on finite chains without decaying channels.
  It costs at most two extra bond indices over `irrep_mpo`'s minimum, one per padded identity channel.
* OpSum does not depend on MPSKit; BlockTensorKit is the shared layer.

---

## Where the current design is under pressure

* **Stage 2 fixes the basis.**
  Distributing composite operands into matrix-unit letters is what costs the 3–4× bond dimension in quantum chemistry ([#38](https://github.com/lkdvos/OpSum.jl/issues/38)).
  operator-bases.md makes the basis explicit, used by `project` in stage 1 and by `instantiate` in stage 6.
* **Stage 5's tie-break.**
  König from the unmatched left vertices prefers covering the prefix side, which can split the switch of a two-index channel set over two sites (operator-bases.md, appendix).
* **Stage 6's block granularity.**
  One level per bond index gives many small blocks; grouping by charge, point group and interval could cut the block count 5× for 1.4× the storage (operator-bases.md, appendix).
* **Throughput of stages 2–4 at `N⁴` terms.**
  Every term is a `couple` call with its own allocations, followed by one global sort; there is no bulk-ingestion path.
