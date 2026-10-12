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

See `benchmark/FIGURES.md` for how the checked-in figures under `docs/src/assets/` are refreshed and what machine/commit the current ones came from.

## Architecture

OpSum.jl converts sums of symmetric quantum operators (e.g. Hamiltonians) into efficient, symmetry-reduced matrix-product-operator (MPO) representations.
The pipeline is: symbolic term algebra → flat term list → per-bond-sector bipartite/SVD compression → reduced MPO tensors.

0. **Lattices** — `src/operators/lattices.jl`
   - `AbstractLattice`, with `FiniteChain(bases)` / `FiniteChain(B, N)` and `InfiniteChain(bases)` / `InfiniteChain(B)`: one operator `Basis` per site, which the letters of a term on that site index into; mixed sites are `vcat(B₁, B₂, …)`.
     Every site is a `Basis`, all of one symmetry (`_checksites`); a lattice of bare spaces throws (the public matrix-unit alphabet went in step 7 of `research/operator-bases.md`).
     Whatever needs the physical space reads it through `_space(site)`, and whatever interprets a letter calls `instantiate(letter, site)`, so the two kinds of site share every code path.
     They differ in exactly two observable ways: whether `getindex` wraps, and whether a site outside `1:length` is an error.
   - A lattice is supplied **where terms are added to an `OperatorSum`**, never where they are written (`Terms`, `couple`, `dot` and placement stay latticeless). Three tiers:
     the term algebra needs only site indices and charges; the whole sweep needs only `N` (`ITOTermTable` reads `length(lat)` and nothing else — no `ElementarySpace` appears anywhere in `irrepgraph*.jl`); only letter validation, `adjoint`, `instantiate` and `irrep_mpo_tensors` need the *sites*.
   - So `OperatorSum` (below) is the one place operators and sites are confronted (`_checklattice`, run when terms *enter*, checks each letter exists in its site's basis). Any iterable of one basis (or space) per site is accepted and converted by `_tolattice`.

1. **Symbolic operator algebra** — `src/operators/algebra/`
   - `SiteOperator{I}` (`siteoperator.jl`): an operator on one site, as a small ordered map from alphabet letter to `ComplexF64` coefficient (two parallel vectors).
     Consume it with `pairs(op)`.
     The bare identity is not a separate case: it is the `passthrough` letter `(unit, 1)` — element 1 of every alphabet — so `scalarop(c, I)` is `c · passthrough` and consumers branch on `ispassthrough`.
   - `OperatorBasis`: supertype for concrete operator alphabets.
   - `Basis` (`basis.jl`, design note `research/operator-bases.md`): an explicit on-site basis, built from neutral operators (two-site ones are factorised into charged pieces, `_factorise_pair`), identity always element 1.
     `project(O, B)` expresses an operator in it (one site → `SiteOperator`, `K` sites → `LocalOperator`), and a letter `(c, n)` is then the `n`-th element of charge `c` in `B`.
     Nothing records which alphabet a letter came from; only `instantiate(letter, B)` interprets it, and the `_instantiate_terms` oracle accepts a `Basis` per site in place of a space.
     The matrix-unit alphabet (`instances(IrrepOperator, V)`, `instantiate(letter, V)`, `_project_units(h)`) is internal only: it is the coordinate system `Basis` is defined and projected through (trivial-charge letters start at `n = 2`).
   - **Stock bases and builders** (`stockbases.jl`, design note §3.7): `spin_basis`, `fermion_basis`, `boson_basis`, `hubbard_basis`, `tj_basis` are memoised functions of the symmetry on TensorKitTensors' spaces, defined from an ordered list of its neutral operators (an operator the symmetry does not admit, or an element already spanned, is skipped: `_basis_from_elements(…; skipdependent = true)`) and completed with matrix units, so products and `H'` never leave them. `spin_ops`, `fermion_ops`, `boson_ops`, `hubbard_ops`, `tj_ops` return every neutral TensorKitTensors operator of the site projected into it, named after its TensorKitTensors function. `Basis(ops; complete = true)` completes a hand-made basis.
   - `IrrepOperator{I}` (`irreptensoroperators.jl`): the irreducible-tensor-operator (ITO) alphabet, `A = IrrepOperator{I}`.
     The fusion-resolved global algebra lives in `irrepalgebra.jl` as three plain immutable structs:
   - `Term{I}` — one term: `sites` (ascending), one `ITOKey` each, and a `coeff`.
     No fusion tree is stored: for a caterpillar the per-position running bond charges *are* the tree (`bondcharges`/`vertexlabels`, inverted by `_tree_from_bonds`), so `tree(t)` derives it.
     `irrepkey.jl` has the full argument, including why `total` alone would not do.
     `==`/`hash` ignore the coefficient, which is what makes summing coincident terms well defined.
     Never mutated, so terms freely share their vectors.
   - `Terms{I}` — a *bag* of them, and **the compressible operator**: what `op[site]`, `couple`, `dot` and `project` return, and what `+`/`-`/`*`/`/` combine.
     It carries **no lattice**, by design: nothing in the term algebra needs a physical space, and the sweep needs only the site *count* (see `Lattices` above and `OperatorSum` below).
     Accumulated by `opsum(terms...)`, which makes **one pass** over each argument (iterables nest); `h + t` and `append!(h, t)` also work, but `+` copies, so folding it is quadratic — `opsum`/`append!` are the linear routes.
     Nothing is deduplicated on the way in: `canonicalize!` takes the normal form, and everything that observes the term set (`length`, iteration, `≈`, `show`) goes through it.
   - **`couple` is order-free.**
     Storage is site-ordered, so an operand acting to the left of an earlier one has its leg *inserted* by `_couple_terms`: the running bond charges from the insertion point rightwards are recomputed (forced, by unique fusion) and the coefficient picks up `Rsymbol(c_a, c_b, c_a ⊗ c_b)` per leg it braids past — for fermionic sectors, exactly the anticommutation sign the caller used to hand-write.
     Coupling two operands of fermion-odd total charge also multiplies the coefficient by `-1` (the graded tensor product), so `couple(cd[i], c[j])` is the physical `c†ᵢcⱼ`; the letters themselves are unchanged.
     That fast path is gated on `_canreorder(I)` = `UniqueFusion` **and** `SymmetricBraiding`. Under a non-abelian symmetry the leg is appended in written order and the caterpillar re-coupled into site order by `_recouple` (`products.jl`): the written tree as a charge-only tensor `C₁ ⊗ … ⊗ C_K ← C_tot` (`_treetensor`, the tensor `_instantiate_term` fuses the letters' charge legs with), legs permuted by TensorKit, read back in the caterpillar basis (`_expand`), memoised per shape — one term can become several. Gated on `_canrecouple(I)` (multiplicity-free, symmetric braiding).
     `dot` accepts either order for *any* symmetry (two legs to the unit sector need no F-move) and inserts the same scalar R-symbol — which is `-1` for odd fermionic charges and for half-integer SU(2) charges.
   - **Products** (`products.jl`, design note `research/operator-bases.md` §3.5): `a * b` on placed operators is the operator product in written order. Every product is first written as one caterpillar in written order with coefficient `cₐ c_b (-1)^{p(a)p(b)}` (the graded sign `couple` uses); `_written!` then turns it into `Term`s (distinct sites; out of order: one R-symbol per swapped pair under an abelian symmetry, `_recouple` under a non-abelian one) or a `PendingTerm` (a site repeats).
     Two caterpillars are concatenated by `_concat`: the second's running charges continue from the first's total when that is neutral (or the symmetry abelian), otherwise the two trees are fused to the channel by the same charge-tensor route (an F-move).
     `*` needs a forced channel (`_forcedproduct`: the totals fuse to one sector), which neutral factors always have.
     Placement on sites that are not strictly increasing goes through the same `_written!` (`hop[i, i]` is an on-site product, `hop[i + 1, i]` a reordering).
     A bag with pending products has no normal form: `canonicalize!` (and so `length`, iteration, `≈`) throws; `show` says so. `opsum!` resolves them with the lattice (`_resolve`): `_written_block` multiplies the letters' embedded tensors in written order, fuses their charge legs by the written caterpillar, and `_project_local` projects back into the sites' bases, memoised per shape. A product outside a basis' span throws there.
   - **Unplaced `couple`/`dot`/`couple_channels`** (`localoperator.jl`): `SiteOperator`/`LocalOperator` operands give a `LocalOperator` on `Σ nsites` slots, by lowering onto consecutive slots and calling the *placed* implementation (no reordering, no R-symbol). Later operands must be single-slot.
   - **Pass-through slots.** A slot a term does not act on (`B + α`, a scalar part of an operand) holds the `passthrough` letter, whose `bond` is the running charge; placement drops those keys, so placed `Term`s never contain one.

2. **Projection (numeric → symbolic)** — `src/operators/algebra/irrepprojection.jl`, `localoperator.jl`
   - `project(h, B)` / `project(h, [B₁, …, B_K])` (`basis.jl`) expresses an operator in bases, by changing coordinates from the internal exact matrix-unit expansion `_project_units(h)`: expand a symmetric `K`-site `TensorMap` (`V₁⊗…⊗V_K ← V₁⊗…⊗V_K`, optionally with a trailing `Vect[I](tot=>1)` charge leg) in the ITO term basis, returning an **unplaced** `LocalOperator{I}` — a `Terms` bag on the relative slots `1:K` plus `nsites = K` (stored explicitly so an empty operator remembers `K`; every term has `sites == 1:K`, checked in the constructor).
     Placement is indexing: `B[i]` is contiguous (`i:i+K-1`), `B[s₁, …, s_K]` explicit and strictly increasing; it relabels each term's `sites` and nothing else, so it is exact for any symmetry — a gap site is a pass-through the sweep reconstructs, and `test/test_local_operator.jl` pins the fermionic (Jordan–Wigner) sign across a gap against sector-resolved spectra.
     `project(O, B)` is the single-site form, returning a `SiteOperator` (a charged `V ← V ⊗ Vect[I](c => 1)` projects too, onto the basis' charged elements), and `LocalOperator(::SiteOperator)` lifts one to `K = 1` (a pass-through part becomes a pass-through slot).
     `LocalOperator` mirrors `Terms`' arithmetic (`+`/`-` need equal `nsites`, scalar `*`/`/`, `B ± α`, `≈`) by delegating to the bag; `SiteOperator` stays separate because it is the entry type of the sweep's bond matrices.
   - **Hermitian conjugation** — `_adjoint_terms(h, lat)` (`builders.jl`; public as `H'` on an `OperatorSum`) goes *through* the forward map: materialise the term's `K`-site block, adjoint it, `project` it back (each slot into its site's basis, `_project_local`; a basis not closed under adjoints throws), rather than symbolically, because a letter's adjoint carries the dual charge and re-expressing the caterpillar coupler over dual charges is B-symbol data.
     Correct by construction (`project` checks its own faithfulness) and memoised per `(keys, spaces)`, so it costs one projection per distinct term *shape*.
     Requires total charge `unit(I)`.
     It is the one term-level operation that genuinely needs the physical spaces, hence it lives on the lattice-carrying container; there is no postfix `h'` on a bare `Terms` (that method throws, pointing at `OperatorSum`).
   - In matrix units (`_project_units`) every term has full support on all `K` sites (an identity factor is a trivial-charge letter); in a `Basis` an identity factor is element 1, the pass-through, so `project(h, B)` can give shorter terms (a `K = 0` constant, a one-site term from `X ⊗ 𝟙`).

3. **Normal form + flat term storage** — `src/operators/compression/irreptermtable.jl`
   - `canonicalize!(ts)`: the single place the normal form is taken — sort the terms, sum coincident ones, drop cancelled ones, **in place**.
     It assumes nothing about its input (no flag records whether a list is already normalised, so nothing can fall out of step with the terms) and is idempotent, just not free the second time — which is why the sweep takes the normal form once, at the `ITOTermTable` boundary.
     Everything that observes the term set (`length`, iteration, `≈`, `==`, `show`) goes through it, so an append count is never mistaken for the number of terms; `nterms_raw(ts)` is the raw count, for tests.
   - `ITOTermTable{I}(ts, N)`: the canonical list materialised as the `K×M` matrices the sweeps index into, plus `nvertices`; idle sites reconstruct the pass-through symbol's running bond charge via `_op_at_ito`.
     The `ITOKey` alphabet and caterpillar fusion helpers live in `irrepkey.jl`.
     Prefix/suffix *classes* of the columns are interned by `_prefix_ids`/`_suffix_ids` (`irrepgraph.jl`), shared by every sweep.

4. **Compression primitive** — `src/datastructures/bipartite.jl`
   - `min_vertex_cover_bipartite` (Hopcroft–Karp maximum matching + König): chooses each bond's basis, fed a bipartite (prefix, suffix) graph per bond-sector.
     Adjacency-list-driven and `O(E√V)`; the dense-matrix method is a convenience wrapper for callers holding an adjacency matrix.

5. **Persistent-graph sweep** — `src/operators/compression/irrepgraph.jl` (the core `ITOGraph`/`_at_site!` sweep), split across `irrepinterning.jl` (suffix/prefix interning), `irrepgraph_vc.jl` (the default `VertexCover` backend) and `irrepgraph_svd.jl` (the `SequentialSVD`/`IndependentSVD` backends)
   - `_irrep_graph_sweep` (the default backend) walks an `ITOGraph` site by site via `_at_site!`.
     The site step's five phases are shared; the bond-basis choice is the pluggable `BondStrategy` (`_bond_basis!`).
     Right vertices are suffix classes, identified by an interned `(sufid, running bond charge)` signature (`_suffix_ids`) rather than a materialised path, and inserted lazily at each term's first active site — the still-pending terms ride a single sentinel on the identity/start channel.
     Cost is `Θ(M·K) + Θ(Σ_terms span)`: linear in `N` for finite-range models.
     `research/persistent-graph-mpo.md` §2 is the design note; §2.2 documents the pending↔started suffix-class **collision**, the one invariant a change here is likely to break (`test/test_irrep_graph.jl` guards it in all three sectors).

6. **The lattice-carrying container and MPO construction** — `src/operators/operatorsum.jl`, `src/operators/compression/symbolicmpo.jl`, `src/operators/compression/irrepmpo.jl`
   - `OperatorSum{I, L}(lat)`: a lattice, a `Terms{I}` bag and an `ExpSum{I}` of `expterm` channels; the one value every consumer takes, so all four finite/infinite × plain/decaying cells share one signature. Fill it with `opsum!`/`opsum(lat, args...)` (linear); `H + x` copies. The latticeless `opsum(args...)` still returns a `Terms`.
   - Validation is on insertion (`_checkinsert`: sector, letters, site range, and on an `InfiniteChain` neutrality and no `K = 0`); one representative per translation class is checked in `unitcell_terms` when the MPO is formed. `H'` goes through `_adjoint_terms` (throws with channels); `instantiate(H)` is finite-only. `mpo(::Terms/Term/ExpSum, …)` throws, pointing at the container.
   - `mpo(H[, alg])`: the symmetric reduced MPO via the per-bond-*sector* sweep over an `ITOTermTable`, as a `SymbolicMPO{I, L}` on either lattice: the lattice (its bases), per site a sparse matrix of `SiteOperator`s (`W.Ws`, `W[i]`), per bond the charges (`W.bondsectors`) and the two identity-channel indices `start`/`finish` (empty when unknown: decaying channels on a finite chain).
     A `SymbolicMPO` iterates over its sites; there is no `(Ws, bondsectors)` destructuring (`reduced(W)` in `test/testutils.jl` is the test helper). The finite compression is `_compress` (irrepmpo.jl), the infinite one `_infinite_window`.
     On a finite chain without channels it goes through `_irrep_channels`, i.e. the finish channel is forced into the cover (`_force_finish!`) so that the result has a Jordan form; on every showcase model that costs nothing over the unconstrained cover, but a constant (`K = 0`) term next to two-site terms costs one index at the first bond (`test/test_irrep_graph.jl` pins both covers).
     `alg` is `BipartiteAlgorithm()` (default) or `SVDBondAlgorithm(trunc; sweep)`; each names a `BondStrategy` (`src/operators/compression/algorithms.jl`) that `_irrep_sweep` dispatches on — `VertexCover` (min-vertex-cover, the default), `IndependentSVD` (`truncrank(k)` = k per bond; the `SVDBondAlgorithm` default) or `SequentialSVD` (k after upstream truncation).
     A *truncating* SVD on a lattice of bases compresses in the basis' Hilbert–Schmidt metric (`metric.jl`: terms rewritten over per-charge orthonormalised letters, `G_c = A'A`, entries mapped back with `A⁻¹`), so the result depends on the span of the bases, not on how they were written; lossless SVD and sites given by spaces skip it.
   - The two verification helpers are public and live here too: `islossless(H[, alg])` is `mpo_terms(mpo(H)) ≈ chain_terms(H)` on a finite chain (infinite chains are unsupported), and `mpo_tensormap(Ts)` contracts a chain of site tensors into `instantiate`'s convention.
   - `mpo_terms(Ws, secs)` reconstructs the operator (faithfulness check) and takes **no** lattice: the bond data names charges but not spaces, and the result is a latticeless `Terms` bag. `irrep_mpo_tensors(W)` (or the reduced-data form `irrep_mpo_tensors(Ws, secs, lat)`) assembles the symmetric `TensorMap`s (one contraction per distinct on-site letter per site, not per bond entry).

7. **Jordan-form emission** — `src/operators/compression/jordanmpo.jl`
   - `jordan_mpo_tensors(H[, alg])` = `jordan_mpo_tensors(mpo(H, alg))`: the same compressed MPO as `irrep_mpo_tensors`, but emitted as one `BlockTensorKit.SparseBlockTensorMap` per site — one *level* per bond index — with the bond indices reordered `(start channel, everything else, finish channel)` and identity at `(1,1)` / `(end,end)`.
     This is the shape MPSKit's `JordanMPOTensor` / `FiniteMPOHamiltonian` consume; MPSKit is only a weak dependency (BlockTensorKit is the shared layer), used by the extension below.
   - The two identity channels come from the sweep (`_irrep_channels`, `g.startidx` / `g.finishidx`), and are *padded* at bonds where the cover spent no index on them.
     The emitted MPO is therefore minimal among Jordan-form MPOs, `≤ +2` per bond over the cover's own minimum (`mpo(H).bondsectors`) — in practice `+1` at the first internal bond and `+1` at the last, since both padded channels are pure identity chains that cannot change the operator.
     The one thing that *does* change the cover is `_force_finish!`, which `mpo` asks for on a finite chain without channels.
   - The infinite cell has a Jordan form too: every bond carries both identity channels, so each bond is reordered by its own `start`/`finish` and nothing is padded (`_jordan_tensors` is shared). A finite chain with decaying channels has none yet.
   - Diagonal unit pass-throughs are emitted as `TensorKit.BraidingTensor`s, which is how a consumer keeps them out of dense storage — and, for a fermionic bond charge crossing a site, is what carries the sign.

7b. **The MPSKit extension** — `ext/OpSumMPSKitExt.jl`
   - `instantiate(W)` calls `_mpskit_hamiltonian(W)`, whose methods come from the extension (`FiniteMPOHamiltonian`/`InfiniteMPOHamiltonian` from the Jordan tensors, through `undef` + `setindex!`, which the boundary tensors need); without MPSKit the fallback throws and says to load it.
   - `research/n2_basis_acceptance.jl` reproduces the #38 numbers of `research/operator-bases.md` §1 from the test environment; `research/n2_spinfree_acceptance.jl` builds the same Hamiltonians spin-free (`E[p, q] * E[r, s]`) and checks them against an exact Fock-space spectrum. The matrix-unit terms of `opsum_qchem_su2.jl` (MPSKitExperimental) have the wrong sign on `a†_p a_q` under the current conventions, so they are a reference for term sets and bond dimensions only, not for energies.

8. **Infinite chains** — `src/operators/infinite/infinitechain.jl`, `infinitegraph.jl`
   - `mpo(H)` on an `OperatorSum` over an `InfiniteChain(bases)` represents `Σ_n translate(H, n·L)`, so `H` is a *generating set* (one representative per translation class — `unitcell_terms(H)` enforces that when the MPO is formed; charge neutrality is checked on insertion).
     It unrolls a window, runs the **unchanged** sweep and searches for the first cell that has reached the translation-invariant fixed point (`_infinite_window`/`_fixedpoint_cell`), returning the cell as a `SymbolicMPO` with the two identity-channel indices.
     `research/infinite-mpo.md` is the design note; §3's canonicalisation (`_rel_suffix_ids`/`_rdesc`, `_canonicalise_rights!`/`_canonicalise_bond!`) is always-on and shared with the finite path — without it the cell does not close.

9. **Exponentially decaying interactions** — `src/operators/infinite/expterms.jl`
   - `expterm(t::Terms; decay = λ, exitsite, string)` stretches the gap before `t`'s exit block geometrically: one *representative* term (which already carries the caterpillar tree, so all fusion channels are named) stands for `Σ λ^{#string sites} …`.
     The `ExpSum` goes into an `OperatorSum` next to the finite-range terms (`opsum(lat, h, expterm(h; decay = λ))`); `mpo` accepts it on either lattice: an `InfiniteChain` (period `L`) or a `FiniteChain` (period 1, i.e. the geometric sum truncated to the chain — `chain_terms(H)` spells it out). A bare `Terms + ExpSum` throws, pointing at the container.
   - Each channel lowers to a small weighted automaton (`_lower_channels`) whose states *are* its suffix classes, with the cyclic core named by an interned loop descriptor; the sweep then treats it as an ordinary right vertex (`ITOGraph.rchan`/`rstate`) whose diagonal is `λ·pass-through`.
     Two things are load-bearing: cyclic states are **forced** into the vertex cover (`_forced_cover`, else λ accumulates instead of landing on the diagonal), and λ must ride as an *edge weight* so the emitted entry carries a letter — tensor assembly iterates `pairs(op)`, so a letter-less `SiteOperator` is silently dropped.
     `research/infinite-mpo.md` §7 is the design note, including what it costs (a channel's period is part of its declaration, so a bigger cell is not free).
   - The `string` is a `SiteOperator` in the sites' basis. A channel's letters land on sites it does not name (translates move by one site on a finite chain; a string fills every gap site), so `_checkchannels` (on insertion) requires those sites to share one basis and the letters to exist in it.

### Key design patterns

- **Concrete types, not expression trees**: there is no lazy symbolic algebra, with one exception.
  `SiteOperator` is a flat letter→coefficient map. The exception is the on-site product: `a[i] * b[i]` (or `hop[i, i]`) is a `PendingTerm` in the bag's `pending` vector, resolved when it enters an `OperatorSum` (see products below).
- **Invariant checks are real `throw`s, never `@assert`**: the sector-purity / cover-validity checks guard *silently wrong output*, and `@assert` is strippable.
  They go through the `@noinline _invariant` helper so the cost is one never-taken branch.
- **`VectorInterface` integration**: the algebra types implement `VectorInterface` norms/inner products for truncation/compression.
- **Instantiation**: `instantiate(op, V)` materializes symbolic ITOs into `TensorMap`s; `instantiate(H::OperatorSum)` is the correctness oracle in tests (`_instantiate_terms(ts, spaces)` is the internal latticeless entry point), kept separate in `src/operators/algebra/irrepinstantiate.jl` since it's needed for verification only, never to build or combine terms.
  `project` is its inverse.
- **Faithfulness is `≈`**: `islossless(H)` — i.e. `mpo_terms(mpo(H)) ≈ chain_terms(H)` (finite chains) — compares canonical term *sets* exactly and coefficients approximately. No lattice is involved in the comparison, which is why the reconstructed bag compares equal.
- **`couple` distributes**: both operands may be composite (several terms, e.g. from `project`); pairs whose charges cannot fuse to `to` are dropped, and it is an error if none do.
  `dot` does *not* distribute — its `-√dim(c)` factor is per-letter.
- **Build local operators wherever reads best.**
  The stock bases and builders are memoised; so is `adjoint` per term shape. Model code uses neutral operators only (builders, `project(O, B)`), combined with `*` and placement; charged letters appear only inside a basis and in tests of the algebra (`test/testutils.jl`: `unitbasis`, `spinvector`, `matrixunit`, `fermionmode`).
  Model code should never mention a bare `IrrepOperator(c, n)` — `n` follows TensorKit's block order, so hard-coding it is a latent bug.
- **`couple` defaults `to` to `unit(I)`** (what a Hamiltonian term needs) and accepts a variadic form for abelian sectors (`FusionStyle(I) isa UniqueFusion`), where every intermediate caterpillar charge is forced by the charges: `couple(cd[1], c[2], cd[3], c[4])`.
  Non-abelian sectors must nest to name each channel — the variadic form throws.
- **Sparse bond matrices**: each sweep accumulates bond entries into a dict-of-keys `Dictionary{CartesianIndex{2}, SiteOperator}` and finalizes it to a stdlib `SparseArrays.SparseMatrixCSC` at the end (`sparse_from_dict`/`storedpairs` in `src/utility/linalg.jl`).

### Test structure

- Test files are `test/test_irrep_*.jl`, `test/test_infinite_*.jl`, `test/test_exp_decay.jl`, `test/test_operator_sum.jl`, `test/test_jordan_mpo.jl`, and for bases `test/test_operator_basis.jl`, `test/test_basis_lattice.jl`, `test/test_basis_svd.jl` and `test/test_mpskit_ext.jl`; each is standalone (defines its own reference matrices).
- Tests use `ParallelTestRunner` for parallel execution; individual files can be run directly with `julia --project`.
- MPSKit is a **weak** dependency of the package (the extension) and a test dependency, tracked as a `url` + `rev = "main"` source rather than a local path so the test environment still resolves on CI and elsewhere. TensorKitTensors is a full dependency (the stock bases are built from its operators).
  It is `main` and not the registered release because the registered `JordanMPOTensor` predates its `tensors,scalars` field rework.
  The seam is exercised only through API-level entry points (`JordanMPOTensor`, `jordanmpotensortype`, `FiniteMPOHamiltonian`), never struct fields.
