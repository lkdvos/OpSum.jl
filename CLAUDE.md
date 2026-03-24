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

OpSum.jl converts sums of quantum operators (e.g. Hamiltonians) into efficient tensor network representations (MPOs, TTNOs). The pipeline flows through four layers:

1. **Symbolic operator algebra** — `src/operators/`
   - `LocalOp{T,A}`: A sum type (via `LightSumTypes.@sumtype`) representing operators on a single local Hilbert space. Variants include scalars, basis elements, `Sum`, `Prod`, `Pow`, `Kron`, `Fun`.
   - `GlobalOp{T,A,S}`: Wraps `LocalOp` with explicit site indices. Syntax: `op[site1, site2, ...]`.
   - `OperatorBasis`: Abstract supertype for concrete operator sets (e.g. `PauliOperator`, `FermionOperator`).
   - Arithmetic (`+`, `-`, `*`, `/`) on these types builds symbolic expression trees.

2. **State machine construction** — `src/statemachines/state_machines.jl`
   - `opsum_vertex_operators(terms, topology)`: converts a list of `GlobalOp` terms into per-site "vertex operator" matrices `W[site]` via finite automaton encoding over Tries/DAWGs.
   - `compress_vertex_operators`: SVD-based compression of vertex operators using `MatrixAlgebraKit`.

3. **Data structures** — `src/datastructures/`
   - `Trie{K,V}`: Prefix tree over fixed-length sequences; used to group operator terms sharing common prefixes/suffixes.
   - `SDAWG` / `DAWGDictionary`: Directed Acyclic Word Graph for suffix-compressed storage; enables sharing common tails of operator strings.
   - `BipartiteGraph`: Used for matching/decomposing tensor network bonds.
   - Tree variants (`TreeTrie`, `TreeDAWG`) for TTNOs on non-linear topologies.

4. **Tensor network representations** — `src/statemachines/ttno.jl`
   - `TreeTopology`: BFS-rooted tree structure describing tensor network geometry.
   - `ttno_terms` / `mpo_bond_optimizations`: Build TTNO/MPO bond tensors from vertex operators.
   - `opsum(terms, sites)`: Convenience entry point for the full MPO construction pipeline.

### Key design patterns

- **Sum types via `LightSumTypes`**: `LocalOp` and `GlobalOp` use `@sumtype` for algebraic variants. Pattern match with `@cases`.
- **`VectorInterface` integration**: All symbolic algebra types implement `VectorInterface` norms/inner products for truncation/compression.
- **Instantiation**: `instantiate(op, sites)` materializes symbolic expressions into dense arrays; used in tests to validate tensor network construction.
- **`SparseMatrixDOK`**: Vertex operator matrices are stored as dict-of-keys sparse matrices before compression.

### Test structure

- `test/instantiate.jl`: Shared test setup defining Pauli matrices and test helpers.
- Tests use `ParallelTestRunner` for parallel execution; individual files can be run directly with `julia --project`.
