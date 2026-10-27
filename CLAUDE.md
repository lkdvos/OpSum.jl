# CLAUDE.md

Guidance for Claude Code in this repository. Design history lives in git.

## Commands

No `Manifest.toml` is checked in; instantiate first (`julia --project -e 'using Pkg; Pkg.instantiate()'`).

```bash
julia --project -e 'using Pkg; Pkg.test()'          # all tests (ParallelTestRunner)
julia --project=test test/compression/test_graph.jl  # one file; each is standalone
julia --project=/tmp/runic -e 'using Pkg; Pkg.add("Runic"); using Runic; Runic.format_file("src/file.jl"; inplace=true)'
julia --project=benchmark benchmark/run.jl --sweep ci                       # scaling benchmarks
julia --project=benchmark scripts/plot_benchmarks.jl --run --sweep full --figure all
julia --project=docs docs/make.jl                    # docs (strict cross-references)
```

Runic is not a dependency (CI uses a reusable workflow). `examples/` and `benchmark/` are not run by `Pkg.test`
(except `benchmark/ShowcaseModels.jl`, which `test/test_showcase_models.jl` includes together with `examples/common.jl`).

## Architecture

Sums of symmetric operators → symmetry-reduced MPOs. Pipeline: term algebra → canonical term table → per-bond-sector
vertex-cover (or SVD) compression → `SymbolicMPO` → tensors.

* **Lattices** (`lattices.jl`): `FiniteChain(bases)`/`FiniteChain(B, N)`, `InfiniteChain(bases)`; every site is a
  `Basis` of one symmetry (`_checksites`). Operators are latticeless; the lattice enters only in `OperatorSum`.
  The sweep needs only `N`; letter validation, `adjoint`, `instantiate` and tensor assembly need the sites.
* **Letters and bases** (`algebra/`): a letter `(c, n)` is the `n`-th element of charge `c` of the site's `Basis`;
  `(unit, 1)` is the identity/pass-through (`ispassthrough`). A `Basis` is always complete, compared by identity, and owns its
  dual elements, letter metric and letter-product table; `project(O, B)` takes overlaps with the dual elements
  (`project_units` is the matrix-unit expansion used only in basis construction); `instantiate(letter, B)` is its inverse.
  Stock bases (`spin_basis`, `fermion_basis`, `boson_basis`, `hubbard_basis`, `tj_basis`) and builders
  (`spin_ops`, …, named after TensorKitTensors functions) live in `stockbases.jl` and are memoised.
* **Terms** (`algebra/terms.jl`, `algebra/letters.jl`): `Term` = ascending sites, one `Factor` per site, coefficient; the
  running bond charges of the caterpillar *are* the fusion tree. `==`/`hash` include the coefficient (field-wise `hash`); `support(t)` = (sites, factors) is the key
  of the normal form. `Terms` is in normal form at all times (sorted, coincident terms summed, zeros dropped; every
  constructor and operation restores it, `+` is a merge, reading never mutates); pending products are kept apart.
  `≈` compares canonical terms as sparse vectors; `==` is exact.
* **`couple`/`dot`/`*`** (`algebra/couple.jl`) are order-free. Abelian (`canreorder`): legs inserted with R-symbols (fermionic signs).
  Non-abelian (`_canrecouple`): written in order, re-coupled by `recouple` (`recouple.jl`, TensorKit fusion-tree `permute`/`merge`). Two fermion-odd operands
  pick up the graded sign. `a[i] * b[i]` is a `PendingTerm`, resolved in the sites' bases when added to an
  `OperatorSum` (`resolve`, `products.jl`); a bag with pending products has no normal form. Operands of `couple`/`dot`
  need a charged letter in every term (the identity is an operator, not an operand), placed or unplaced alike.
* **Sweep** (`compression/graph*.jl`): `sweep(TermTable(ts), N, strategy)` → `(; Ws, bondsectors, start, finish)`;
  it walks a `SweepGraph` site by site (`at_site!`; its fields are grouped in `RightVertices`, `LeftVertices`,
  `Pending`, `Boundary`, `Scratch`; right-vertex kinds are the `RightKind` enum, SVD
  sweeps seed eagerly, the cover lazily); the bond basis comes from
  a `BondStrategy` (`VertexCover` default via Hopcroft–Karp + König in `utility/bipartite.jl`; `IndependentSVD`,
  `SequentialSVD`; each `bond_basis!` returns a `BondBasis`, SVD entries below `SVD_ATOL` are dropped). Load-bearing invariant: the pending↔started suffix-class collision (`_promote_pending!`), guarded
  in `test/compression/test_graph.jl` in all three sectors. Truncating SVD works in the bases' Hilbert–Schmidt metric (`metric.jl`).
* **Container and output** (`operatorsum.jl`, `mpo/{symbolicmpo,tensors,jordan}.jl`): `OperatorSum` holds
  lattice + `Terms` + `ExpSum` channels and validates on insertion (`checkinsert`, `checkchannels`). `mpo(H[, alg])`
  → `SymbolicMPO` (`start`/`finish` always have length `N`); on a finite chain the finish channel is forced into the
  cover (`_force_finish!`) so a Jordan form exists, with decaying channels too. `mpo_terms(W)` (finite) inverts it;
  `mpo_tensors(W)`, `jordan_mpo_tensors` (via `jordan_layout`; BlockTensorKit, MPSKit layout) and `instantiate(W)`
  (a method of the MPSKit extension, `ext/`) emit tensors; the `(Ws, bondsectors)` forms are internal
  (`reduced_terms`, `reduced_tensors`). `H'` goes through dense blocks (`adjoint_terms`).
* **Infinite chains** (`infinite/`): `mpo` on an `InfiniteChain` unrolls a window, runs the same sweep and finds the
  translation-invariant cell (`infinite_window`); the suffix canonicalisation (`_canonicalise_rights!`) is required
  for the cell to close. `expterm` channels (`src/expterms.jl`; the automaton is `compression/channels.jl`) lower to weighted automata: cyclic states are forced into
  the cover, and λ rides as an edge weight so every emitted entry carries a letter.
* **Time evolution** (`timeevolution/{kronpower,taylormpo,taylor}.jl`, `algebra/letterproducts.jl`):
  `taylor_mpo(H|W, N; extension = true, compression = false)` → `TaylorMPO`, the symbolic `exp(τH)` MPO through `τ^N`
  (Van Damme et al., SciPost Phys. 17, 135); entries are `TaylorEntry` τ-polynomials, `evaluate(W, τ)` / `W(τ)` /
  `mpo_tensors(W, τ)` fix `τ` (`U ≈ exp(τH)`: imaginary time `τ = -β`, real time `τ = -it`). Abelian only
  (fermions included), non-abelian throws; the MPSKit extension adds `instantiate(W, τ)` and `make_time_mpo(…,
  TaylorCluster)`. On-site products are coefficient arithmetic on per-basis structure constants (`letter_product`,
  memoised per letter pair); the `N`-fold Kronecker power of the Jordan form is enumerated depth-first with a word trie
  so each distinct ordered product is computed once. Load-bearing: (1) the sign rule, each entry carries
  `∏_{k<m} R(q_k, bR_m)` (the `R(q_k, q_m)` part cancels the graded sign of the on-site product, the rest reorders
  letter legs past bond legs; invisible for bosons, wrong for fermions); (2) pass order extension (Alg 3, `H^{N+1}`
  entries evaluated lazily, never materialised) → fold (Alg 1) → merge (Alg 2, exact) → compression (Alg 4, lossy at
  `τ^{N+1}`) → orphan pruning, and the merge orientation (1's, start channels, to the front, 3's, finish channels, to
  the back, matching MPSKit's `TaylorCluster`). Finite chains close the right boundary with both identity channels.

## Conventions

* Concrete types, no expression trees (on-site products are the one deferred case).
* Invariant checks are `throw`s via `invariant_violation`, never `@assert`.
* Model code uses neutral operators only (builders, `project`); never a hard-coded `Letter(c, n)`.
* Verification: `islossless(H)` (= `mpo_terms(mpo(H)) ≈ chain_terms(H)`) and the dense oracle `instantiate(H)`.
* Tests: `test/**/test_*.jl` (subfolders mirror `src/`, one file per source file; end-to-end ones sit at the top), discovered automatically; shared helpers in `test/testutils.jl` (`unitbasis`,
  `spinvector`, `matrixunit`, `fermionmode`, `physmatrix`, `reduced`). MPSKit is a weak dependency tracked from
  `main` (`url` + `rev` source); TensorKitTensors is a full dependency.
* `benchmark/n2/n2_*_acceptance.jl` reproduce the #38 N₂ numbers (centre bond 102/192 vs 347/677 in matrix units) and
  check the spin-free Hamiltonian against an exact spectrum.
