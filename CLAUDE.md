# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project test/test_irrep_mpo.jl
julia --project test/test_irrep_alphabet.jl
julia --project test/test_irrep_termtable.jl

# Format code (Runic)
julia --project -e 'using Runic; Runic.format_file("src/file.jl")'

# Check formatting without modifying
julia --project -e 'using Runic; Runic.format_file("src/file.jl"; check=true)'
```

## Architecture

OpSum.jl converts sums of symmetric quantum operators (e.g. Hamiltonians) into efficient, symmetry-reduced matrix-product-operator (MPO) representations. The pipeline is: symbolic term algebra → flat term list → per-bond-sector bipartite/SVD compression → reduced MPO tensors.

(A dense/Pauli pipeline based on a `GlobalOp` expression tree previously ran in parallel; it has been removed — only the symmetric ITO track remains.)

1. **Symbolic operator algebra** — `src/operators/`
   - `LocalOp{T,A}`: a sum type (via `LightSumTypes.@sumtype`) for operators on one local Hilbert space; variants are scalars, basis elements, `Sum`, `Prod`, `Pow`, `Kron`, `Fun`. `A` is the on-site alphabet.
   - `OperatorBasis`: supertype for concrete operator alphabets.
   - `IrrepOperator{I}` (`irreptensoroperators.jl`): the irreducible-tensor-operator (ITO) alphabet, `A = IrrepOperator{I}`. The fusion-resolved global algebra is the term-sum `TermSum`/`TermKey` (`irrepalgebra.jl`), built by `op[site]`, `+`, `scale`, and `couple`/`dot`.

2. **Projection (numeric → symbolic)** — `src/operators/irrepprojection.jl`
   - `project(h, sites)`: expand a symmetric `K`-site `TensorMap` (`V₁⊗…⊗V_K ← V₁⊗…⊗V_K`, optionally with a trailing `Vect[I](tot=>1)` charge leg) in the ITO term basis, returning a `TermSum`. `project(O, V)` is the single-site form, returning a `LocalOp`. This is the inverse of `instantiate` and the intended way to write operators down — hard-coding letter indices `(c, n)` is fragile because `n` follows TensorKit's block order.
   - The candidate basis `(ops, tree)` is orthogonal and complete, with the closed-form diagonal `inner(E,E) = dim(tot) / Π_k dim(c_k)`, so coefficients are plain inner products — no solve. Coefficients below tolerance are dropped and the result is re-materialized and checked against the input (throws if unfaithful).
   - `matrixunit(V, out, in)`: `|out⟩⟨in|` as a `LocalOp`, for abelian/fermionic spaces.
   - Every projected term has full support on all `K` sites: an on-site identity factor appears as a trivial-charge letter, not a shorter term.

3. **Flat term storage** — `src/operators/`
   - `ITOTermTable{I}` (`irreptermtable.jl`): flat, sparse-per-term storage of a `TermSum` — each term's active `(site, ITOKey)` factors in `K×M` matrices plus a `coeffs` vector; idle sites reconstruct the pass-through symbol's running bond charge via `_op_at_ito`. The `ITOKey` alphabet and caterpillar fusion helpers live in `irrepkey.jl`.

4. **Compression primitive** — `src/datastructures/bipartite.jl`
   - `min_vertex_cover_bipartite` (Hopcroft–Karp maximum matching + König): chooses each bond's basis, fed a bipartite (prefix, suffix) graph per bond-sector.

5. **MPO construction** — `src/operators/irrepmpo.jl`
   - `irrep_mpo(H::TermSum, sites[, alg])`: symmetric reduced MPO from a `TermSum` via the per-bond-*sector* sweep (`_irrep_bipartite`) over an `ITOTermTable`; returns reduced bond matrices + per-bond charge sectors. `alg` is `BipartiteAlgorithm()` (default, min-vertex-cover) or `SVDBondAlgorithm()`.
   - `mpo_terms` reconstructs the `TermSum` (faithfulness check) and `irrep_mpo_tensors` assembles the symmetric `TensorMap`s.

### Key design patterns

- **Sum types via `LightSumTypes`**: `LocalOp` uses `@sumtype`; pattern-match with `@cases`.
- **`VectorInterface` integration**: symbolic algebra types implement `VectorInterface` norms/inner products for truncation/compression.
- **Instantiation**: `instantiate(op, V)` materializes symbolic ITOs into `TensorMap`s; `instantiate(ts::TermSum, sites)` is the correctness oracle in tests. `project` is its inverse.
- **`couple` distributes**: both operands may be composite (several terms, e.g. from `project`); pairs whose charges cannot fuse to `to` are dropped, and it is an error if none do. `dot` does *not* distribute — its `-√dim(c)` factor is per-letter.
- **Sparse bond matrices**: each sweep accumulates bond entries into a dict-of-keys `Dictionary{CartesianIndex{2}, LocalOp}` and finalizes it to a stdlib `SparseArrays.SparseMatrixCSC` at the end (`sparse_from_dict`/`storedpairs` in `src/utility/linalg.jl`).

### Test structure

- Test files are `test/test_irrep_*.jl`; each is standalone (defines its own reference matrices).
- Tests use `ParallelTestRunner` for parallel execution; individual files can be run directly with `julia --project`.
