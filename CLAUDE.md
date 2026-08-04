# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

No `Manifest.toml` is checked in (it is gitignored), so instantiate first:

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run a single test file (uses the test environment)
julia --project=test test/test_irrep_mpo.jl
julia --project=test test/test_irrep_graph.jl
julia --project=test test/test_bipartite.jl

# Format code (Runic). Runic is not a project dependency — CI uses a reusable workflow — so install
# it into a throwaway environment rather than adding it to Project.toml.
julia --project=/tmp/runic -e 'using Pkg; Pkg.add("Runic"); using Runic; Runic.format_file("src/file.jl", "src/file.jl"; inplace=true)'
```

```bash
# Scaling benchmarks + figures (separate environment: BenchmarkTools, CairoMakie, JSON3)
julia --project=benchmark benchmark/run.jl --sweep ci
julia --project=benchmark scripts/plot_benchmarks.jl --run --sweep full --figure all
```

See `benchmark/FIGURES.md` for how the checked-in figures under `docs/src/assets/` are refreshed and
what machine/commit the current ones came from.

## Architecture

OpSum.jl converts sums of symmetric quantum operators (e.g. Hamiltonians) into efficient, symmetry-reduced matrix-product-operator (MPO) representations. The pipeline is: symbolic term algebra → flat term list → per-bond-sector bipartite/SVD compression → reduced MPO tensors.

(A dense/Pauli pipeline based on a `GlobalOp` expression tree previously ran in parallel; it has been removed — only the symmetric ITO track remains.)

1. **Symbolic operator algebra** — `src/operators/`
   - `OnsiteOp{I}` (`onsiteop.jl`): an operator on one site, as a small ordered map from alphabet letter to `ComplexF64` coefficient (two parallel vectors). Consume it with `pairs(op)`. The bare identity is not a separate case: it is the `passthrough` sentinel letter, so `scalarop(c, I)` is `c · passthrough` and consumers branch on `ispassthrough`.
   - `OperatorBasis`: supertype for concrete operator alphabets.
   - `IrrepOperator{I}` (`irreptensoroperators.jl`): the irreducible-tensor-operator (ITO) alphabet, `A = IrrepOperator{I}`. The fusion-resolved global algebra is `TermSum` (`= TermList`, `irrepalgebra.jl`), built by `op[site]`, `+`, `scale`, and `couple`/`dot`.
   - `TermSum{I}` is an **append-only column store**: `K × M` flat arrays of `(site, ITOKey)` sharing a growable `TermBuffer`, plus `coeffs` and an optional `lattice`. Every operation appends — `+` concatenates (in place when the left operand is its buffer's live tip; see THE ONE INVARIANT in `irrepalgebra.jl`), `couple` copies a column and pushes one more slot. Storing running bond charges rather than a `FusionTree` is what makes that possible: for a caterpillar, `ITOKey.bond`/`.vertex` per position *are* `bondcharges(tree)`/`vertexlabels(tree)`, and `_tree_from_bonds` inverts it.
   - `TermKey` is the key type of the **canonical view** `H.terms` (`termdict`), not storage. `length(H)`, `keys(H)`, `pairs(H)`, `isempty(H)` and `≈` all go through `canonical`, so a pre-dedup row count is never observable. `nrows(H)` is the raw count, for tests.
   - The lattice travels with the operator: `onlattice(H, sites)` binds it (checking every letter against its site's space and every site against `1:N`), `lattice(H)` reads it back, and `irrep_mpo(H)` / `instantiate(H)` / `jordan_mpo_tensors(H)` then need no second argument. Placement cannot bind it — `A[i]` sees one space and no `N`.

2. **Projection (numeric → symbolic)** — `src/operators/irrepprojection.jl`
   - `project(h, sites)`: expand a symmetric `K`-site `TensorMap` (`V₁⊗…⊗V_K ← V₁⊗…⊗V_K`, optionally with a trailing `Vect[I](tot=>1)` charge leg) in the ITO term basis, returning a `TermSum`. `project(O, V)` is the single-site form, returning an `OnsiteOp`. This is the inverse of `instantiate` and the intended way to write operators down — hard-coding letter indices `(c, n)` is fragile because `n` follows TensorKit's block order.
   - The candidate basis `(ops, tree)` is orthogonal and complete, with the closed-form diagonal `inner(E,E) = dim(tot) / Π_k dim(c_k)`, so coefficients are plain inner products — no solve. Coefficients below tolerance are dropped and the result is re-materialized and checked against the input (throws if unfaithful).
   - `matrixunit(V, out, in)`: `|out⟩⟨in|` as an `OnsiteOp`, for abelian/fermionic spaces.
   - Every projected term has full support on all `K` sites: an on-site identity factor appears as a trivial-charge letter, not a shorter term.

3. **Canonicalisation + flat term storage** — `src/operators/irreptermtable.jl`
   - `canonical(H)`: the single place the normal form is taken — coincident columns summed (hashed in place through `ColumnRef`, no per-term allocation), cancelled ones dropped. Memoised on the term list and lattice-free, so `H.terms`, `irrep_mpo` and `jordan_mpo_tensors` share one pass. Before Stage 2 this normal form was computed *twice*, once by `TermSum`'s dictionary and again here.
   - `ITOTermTable{I}`: the canonical list materialised as the `K×M` matrices the sweeps index into, plus `nvertices`; idle sites reconstruct the pass-through symbol's running bond charge via `_op_at_ito`. The `ITOKey` alphabet and caterpillar fusion helpers live in `irrepkey.jl`. Prefix/suffix *classes* of the columns are interned by `_prefix_ids`/`_suffix_ids` (`irrepgraph.jl`), shared by every sweep.

4. **Compression primitive** — `src/datastructures/bipartite.jl`
   - `min_vertex_cover_bipartite` (Hopcroft–Karp maximum matching + König): chooses each bond's basis, fed a bipartite (prefix, suffix) graph per bond-sector. Adjacency-list-driven and `O(E√V)`; the dense-matrix method is a convenience wrapper for callers holding an adjacency matrix.

5. **Persistent-graph sweep** — `src/operators/irrepgraph.jl`
   - `_irrep_graph_sweep` (the default backend) walks an `ITOGraph` site by site via `_at_site!`. The site step's five phases are shared; the bond-basis choice is the pluggable `BondStrategy` (`_bond_basis!`). Right vertices are suffix classes, identified by an interned `(sufid, running bond charge)` signature (`_suffix_ids`) rather than a materialised path, and inserted lazily at each term's first active site — the still-pending terms ride a single sentinel on the identity/start channel. Cost is `Θ(M·K) + Θ(Σ_terms span)`: linear in `N` for finite-range models. `research/persistent-graph-mpo.md` §2 is the design note; §2.2 documents the pending↔started suffix-class **collision**, the one invariant a change here is likely to break (`test/test_irrep_graph.jl` guards it in all three sectors).

6. **MPO construction** — `src/operators/irrepmpo.jl`
   - `irrep_mpo(H::TermSum[, sites][, alg])`: symmetric reduced MPO from a `TermSum` via the per-bond-*sector* sweep over an `ITOTermTable` (`sites` may be omitted for a lattice-bound `H`); returns reduced bond matrices + per-bond charge sectors. `alg` is `BipartiteAlgorithm()` (default) or `SVDBondAlgorithm(trunc; sweep)`; each names a `BondStrategy` (`src/algorithms.jl`) that `_irrep_sweep` dispatches on — `VertexCover` (min-vertex-cover, the default), `IndependentSVD` (`truncrank(k)` = k per bond; the `SVDBondAlgorithm` default) or `SequentialSVD` (k after upstream truncation).
   - `mpo_terms` reconstructs the `TermSum` (faithfulness check) and `irrep_mpo_tensors` assembles the symmetric `TensorMap`s (one contraction per distinct on-site letter per site, not per bond entry).

7. **Jordan-form emission** — `src/operators/jordanmpo.jl`
   - `jordan_mpo_tensors(H[, sites][, alg])`: the same compressed MPO as `irrep_mpo_tensors`, but emitted as one `BlockTensorKit.SparseBlockTensorMap` per site — one *level* per bond index — with the bond indices reordered `(start channel, everything else, finish channel)` and identity at `(1,1)` / `(end,end)`. This is the shape MPSKit's `JordanMPOTensor` / `FiniteMPOHamiltonian` consume; OpSum does **not** depend on MPSKit (BlockTensorKit is the shared layer, and the only new dependency).
   - The two identity channels come from the sweep (`_irrep_channels`, `g.startidx` / `g.finishidx`), and are *padded* at bonds where the cover spent no index on them. The emitted MPO is therefore minimal among Jordan-form MPOs, `≤ +2` per bond over `irrep_mpo`'s unconstrained minimum — in practice `+1` at the first internal bond and `+1` at the last. `research/persistent-graph-mpo.md` §2.4 has the argument; the one thing that *does* change the cover is `_force_finish!`, which runs only on this path.
   - Diagonal unit pass-throughs are emitted as `TensorKit.BraidingTensor`s, which is how a consumer keeps them out of dense storage — and, for a fermionic bond charge crossing a site, is what carries the sign.

### Key design patterns

- **Concrete types, not expression trees**: there is no lazy symbolic algebra. `OnsiteOp` is a flat letter→coefficient map; on-site products are not symbolic (build the `TensorMap` and `project` it — see `research/onsite-products.md`).
- **Invariant checks are real `throw`s, never `@assert`**: the sector-purity / cover-validity checks guard *silently wrong output*, and `@assert` is strippable. They go through the `@noinline _invariant` helper so the cost is one never-taken branch.
- **`VectorInterface` integration**: the algebra types implement `VectorInterface` norms/inner products for truncation/compression.
- **Instantiation**: `instantiate(op, V)` materializes symbolic ITOs into `TensorMap`s; `instantiate(ts::TermSum[, sites])` is the correctness oracle in tests. `project` is its inverse.
- **Faithfulness is `≈`**: `mpo_terms(irrep_mpo(H, sites)...) ≈ H` compares canonical term *sets* exactly and coefficients approximately; it ignores the lattice, since `mpo_terms` knows the bonds but not the spaces.
- **`couple` distributes**: both operands may be composite (several terms, e.g. from `project`); pairs whose charges cannot fuse to `to` are dropped, and it is an error if none do. `dot` does *not* distribute — its `-√dim(c)` factor is per-letter.
- **`couple` defaults `to` to `unit(I)`** (what a Hamiltonian term needs) and accepts a variadic form for abelian sectors (`FusionStyle(I) isa UniqueFusion`), where every intermediate caterpillar charge is forced by the charges: `couple(cd[1], c[2], cd[3], c[4])`. Non-abelian sectors must nest to name each channel — the variadic form throws.
- **Sparse bond matrices**: each sweep accumulates bond entries into a dict-of-keys `Dictionary{CartesianIndex{2}, OnsiteOp}` and finalizes it to a stdlib `SparseArrays.SparseMatrixCSC` at the end (`sparse_from_dict`/`storedpairs` in `src/utility/linalg.jl`).

### Test structure

- Test files are `test/test_irrep_*.jl` plus `test/test_jordan_mpo.jl`; each is standalone (defines its own reference matrices).
- Tests use `ParallelTestRunner` for parallel execution; individual files can be run directly with `julia --project`.
- MPSKit is a **test-only** dependency, and deliberately the *registered* version, not a path source: the seam is exercised through the surface a released MPSKit provides (`JordanMPOTensor(::SparseBlockTensorMap)`, `jordanmpotensortype`, `FiniteMPOHamiltonian`), so CI can instantiate the test environment.
