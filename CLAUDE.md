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

2. **Flat term storage** — `src/operators/`
   - `ITOTermTable{I}` (`irreptermtable.jl`): flat, sparse-per-term storage of a `TermSum` — each term's active `(site, ITOKey)` factors in `K×M` matrices plus a `coeffs` vector; idle sites reconstruct the pass-through symbol's running bond charge via `_op_at_ito`. The `ITOKey` alphabet and caterpillar fusion helpers live in `irrepkey.jl`.

3. **Compression primitive** — `src/datastructures/bipartite.jl`
   - `min_vertex_cover_bipartite` (Hopcroft–Karp maximum matching + König): chooses each bond's basis, fed a bipartite (prefix, suffix) graph per bond-sector.

4. **MPO construction** — `src/operators/irrepmpo.jl`
   - `irrep_mpo(H::TermSum, sites[, alg])`: symmetric reduced MPO from a `TermSum` via the per-bond-*sector* sweep (`_irrep_bipartite`) over an `ITOTermTable`; returns reduced bond matrices + per-bond charge sectors. `alg` is `BipartiteAlgorithm()` (default, min-vertex-cover) or `SVDBondAlgorithm()`.
   - `mpo_terms` reconstructs the `TermSum` (faithfulness check) and `irrep_mpo_tensors` assembles the symmetric `TensorMap`s.

### Key design patterns

- **Sum types via `LightSumTypes`**: `LocalOp` uses `@sumtype`; pattern-match with `@cases`.
- **`VectorInterface` integration**: symbolic algebra types implement `VectorInterface` norms/inner products for truncation/compression.
- **Instantiation**: `instantiate(op, V)` materializes symbolic ITOs into `TensorMap`s; `instantiate(ts::TermSum, sites)` is the correctness oracle in tests.
- **`SparseMatrixDOK`**: bond matrices are dict-of-keys sparse matrices, finalized to sparse form at the end of each sweep.

### Test structure

- Test files are `test/test_irrep_*.jl`; each is standalone (defines its own reference matrices).
- Tests use `ParallelTestRunner` for parallel execution; individual files can be run directly with `julia --project`.
