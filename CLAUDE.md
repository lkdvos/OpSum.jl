# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
julia --project -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project test/test_trie.jl
julia --project test/test_mpo_bipartite.jl
julia --project test/test_operatoralgebra.jl

# Format code (Runic)
julia --project -e 'using Runic; Runic.format_file("src/file.jl")'

# Check formatting without modifying
julia --project -e 'using Runic; Runic.format_file("src/file.jl"; check=true)'
```

## Architecture

OpSum.jl converts sums of quantum operators (e.g. Hamiltonians) into efficient matrix-product-operator (MPO) representations. There are two parallel pipelines — a dense/Pauli one and a symmetric/irrep (ITO) one — sharing the same shape: symbolic algebra → flat term list → per-bond bipartite compression → MPO.

1. **Symbolic operator algebra** — `src/operators/`
   - `LocalOp{T,A}`: a sum type (via `LightSumTypes.@sumtype`) for operators on one local Hilbert space; variants are scalars, basis elements, `Sum`, `Prod`, `Pow`, `Kron`, `Fun`.
   - `GlobalOp{T,A,S}`: wraps `LocalOp` with explicit site indices — `op[site1, site2, ...]`. Arithmetic (`+`, `-`, `*`) builds symbolic expression trees.
   - `OperatorBasis`: supertype for concrete operator sets; `PauliOperator` (`paulioperators.jl`) is the dense basis.
   - Symmetric track: `IrrepOperator` (`irreptensoroperators.jl`) plus the fusion-resolved term algebra `TermSum`/`TermKey` (`irrepalgebra.jl`), built with `couple`/`dot`.

2. **Flat term storage** — `src/operators/`
   - `TermTable{Op,T}` (`termtable.jl`): flat, sparse-per-term storage — each term's non-identity `(site, op)` factors in `K×M` matrices plus a `coeffs` vector, built directly from a `GlobalOp`. Mirrors ITensorMPOConstruction's `OpIDSum`.
   - `ITOTermTable{I}` (`irreptermtable.jl`): the symmetric counterpart, storing each term's active `(site, ITOKey)` factors; idle sites reconstruct the pass-through symbol's running bond charge via `_op_at_ito`. The `ITOKey` alphabet and caterpillar fusion helpers live in `irrepkey.jl`.

3. **Compression primitive** — `src/datastructures/bipartite.jl`
   - `min_vertex_cover_bipartite` (Hopcroft–Karp maximum matching + König): chooses each bond's basis. Both pipelines feed it a bipartite (prefix, suffix) graph per bond.

4. **MPO construction** — `src/statemachines/graphbuilding.jl`, `src/operators/irrepmpo.jl`
   - `mpo_bond_optimizations(vertices, ex::GlobalOp[, alg])`: dense MPO from a `GlobalOp`. `alg` is `BipartiteAlgorithm()` (default, min-vertex-cover) or `SVDBondAlgorithm()`. Runs a per-bond sweep over a `TermTable`; returns `Vector{SparseMatrixDOK{LocalOp}}`.
   - `irrep_mpo(H::TermSum, sites)`: symmetric reduced MPO from a `TermSum` via the per-bond-*sector* bipartite sweep (`_irrep_bipartite`) over an `ITOTermTable`; returns reduced bond matrices + per-bond charge sectors. `mpo_terms` reconstructs the `TermSum` (faithfulness check) and `irrep_mpo_tensors` assembles the symmetric `TensorMap`s.
   - `mpo_to_dense` (`state_machines.jl`): contracts the bond matrices into a dense operator for validation.

### Key design patterns

- **Sum types via `LightSumTypes`**: `LocalOp` and `GlobalOp` use `@sumtype`; pattern-match with `@cases`.
- **`VectorInterface` integration**: symbolic algebra types implement `VectorInterface` norms/inner products for truncation/compression.
- **Instantiation**: `instantiate(op, sites)` materializes symbolic expressions into dense arrays / `TensorMap`s; used as the correctness oracle in tests.
- **`SparseMatrixDOK`**: bond matrices are dict-of-keys sparse matrices, finalized to sparse form at the end of each sweep.

### Test structure

- `test/instantiate.jl`: Shared test setup defining Pauli matrices and test helpers.
- Tests use `ParallelTestRunner` for parallel execution; individual files can be run directly with `julia --project`.
