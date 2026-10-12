# Handoff: implementing explicit on-site operator bases

For a fresh session that starts implementing the design in [operator-bases.md](operator-bases.md).
Read this file first, then operator-bases.md §3 (the design) and §6 (what is decided and what is open), then `CLAUDE.md`.

---

## 0. State of the world

* Branch `research/operator-bases`, four commits on top of `main` at `5f246be`.
  It contains research notes only, no library code:
  * [operator-bases.md](operator-bases.md): the spec, agreed with the author after a design meeting and refined since.
  * [overview-pipeline.md](overview-pipeline.md), [overview-types.md](overview-types.md), [overview-interface.md](overview-interface.md): the state of `main` before this work, for reference.
* Nothing is pushed.
  Start the implementation on a new branch off this one (or off `main`, cherry-picking the notes).
* A separate experiment lives in the worktree `../tiebreak`, on branch `exp/cover-tiebreak`, based on `c69599b` plus a cherry-pick of `b923b3f`.
  It has **uncommitted** changes: a `prefer = :left | :right` option on `min_vertex_cover_bipartite` and a `COVER_POLICY` switch in `irrepgraph_vc.jl`, whose default is currently set to `:right` for a test run.
  It belongs to the appendix of operator-bases.md, not to this work; leave it alone unless asked.

## 1. What is being built, in one paragraph

The pipeline currently splits every on-site operator into matrix-unit letters inside `couple`, which inflates bond dimensions 2–4× whenever an operator is a sum of matrix units (#38).
The fix is an explicit `Basis`.
A basis is defined from *neutral* operators taken from TensorKitTensors.jl; two-site ones are factorised into charged single-site pieces, which become the basis elements, with the identity always at element 1.
Users project neutral operators into a basis, build sums with `*` over per-site bases (`FiniteChain(bases)` / `InfiniteChain(bases)`), call `mpo(H)` to get a `SymbolicMPO`, and `instantiate(W)` to get MPSKit's `FiniteMPOHamiltonian` / `InfiniteMPOHamiltonian`.
The sweep itself does not change: it only ever compares letters, and a letter becomes `(charge, index into the site's basis)` instead of `(charge, index into matrix units)`.

## 2. Decisions you must not reopen

All of these are recorded in operator-bases.md §3.1 and §6 ("Decided"); the author made each one explicitly.

* No basis tag on letters; a mismatch between an operator's basis and its site's basis is silently wrong, and that is accepted.
* No automatic basis discovery (an `:auto` prototype existed and was deliberately dropped).
* `project` always takes a basis; the basis-free forms go away.
* User-facing operators are always neutral (single-site neutral operators and neutral pairs); charged pieces exist only inside a `Basis`.
* `*` is defined whenever the overall channel is forced, which every product of neutral operators is.
* On-site products are written `a[i] * b[i]` (or a pair placed on one site, `hop[i, i]`) and resolved in the site's basis when the term enters the sum; `(a * b)[i]` is dropped.
* `couple` and `*` accept any site order under any symmetry: non-abelian operands are re-coupled with F- and R-moves (TensorKit's fusion-tree `merge`/`braid`/`Fsymbol`), lifting today's site-order restriction.
* Names: `Basis`, `mpo(H)` (replacing `irrep_mpo`), `SymbolicMPO` (avoiding `FiniteMPO`/`InfiniteMPO`, which MPSKit also exports), `instantiate(W)` returning the MPSKit struct via a package extension.
* Stock bases are memoised functions of the symmetry, named after TensorKitTensors' spaces: `spin_basis`, `fermion_basis`, `hubbard_basis`, `tj_basis`, `boson_basis`. No clock bases for now.
* No separate no-allocation entry for building `N⁴` terms.
* This package plus TensorKitTensors are meant to replace MPSKitModels.jl.

Three consequences were written into the note without an explicit confirmation from the author; check them before building on them:
a hand-made `Basis` is also built from neutral operators only, `hop[i, i]` is an on-site product, and model code no longer needs `couple`.

## 3. Still open

* Where MPSKitModels' lattices and named models (`hubbard_model`, `heisenberg_XXZ`, …) go (operator-bases.md §6).
* ~~The deterministic gauge convention for factorised pieces~~: settled in §7, item 3.

## 4. The plan

Follow operator-bases.md §7; each step should leave `Pkg.test()` green.

1. **`Basis` and `project(O, basis)` on one space, identity at element 1.**
   The riskiest part is replacing the pass-through sentinel (`IrrepOperator(unit, 0)`, `ispassthrough`, `passthrough(I)`) with element 1.
   It is used in `irrepkey.jl`, `siteoperator.jl`, `localoperator.jl`, `irrepalgebra.jl`, `irreptermtable.jl`, `irrepgraph.jl`, `irrepgraph_svd.jl`, `irrepmpo.jl` and `expterms.jl` (about 25 sites); `research/persistent-graph-mpo.md` §2.2 explains the invariant around the start channel that a change here can break, and `test/test_irrep_graph.jl` guards it.
2. **`SymbolicMPO`, `instantiate(entry, basis)`, and the MPSKit extension** (MPSKit as a weak dependency).
   This replaces the hand-written `to_jordan` adapter of `examples/mpskit.jl`.
3. **Per-site bases** (`FiniteChain(bases)`, `InfiniteChain(bases)`, `vcat`).
4. **`*` across sites under an abelian symmetry, then on-site product nodes** resolved on insertion (operator-bases.md §3.5).
5. **Non-abelian re-coupling** in `couple` and `*` (§3.6).
   `test/test_irrep_algebra.jl:318` currently pins that non-abelian reordering is refused; it becomes a test of the re-coupled result against the dense oracle.
6. **Infinite chains, `expterm`, and the SVD backends** (Gram matrix per charge for truncation weights).
7. **Stock bases and builders** (§3.7), starting with spins and the Hubbard site of #38.

## 5. Acceptance numbers

Measured during the design work; an implementation should reproduce them.
Bond dimension is `sum(dim, bondsectors)` at the central bond unless stated.

| model | today (`main`) | with the right basis |
|---|---|---|
| `XX+YY+ZZ` chain, ℂ², `N = 8` | 6 | 5 |
| spinful Hubbard, U₁⊠U₁, `N = 10` | 10 | 6 |
| spinful all-to-all hopping, random `tᵢⱼ`, `N = 10` | 42 | 22 |
| SU(2) Hubbard (`examples/fermions.jl` block), `N = 4` | 10 | 6 |
| N₂/STO-3G, #38 terms, channels / dims | 347 / 677 | 102 / 192 |
| N₂/cc-pVDZ, #38 terms, bond 14, channels / dims | 2984 / 5924 | 786 / 1542 |

For N₂ the "right basis" is the SU(2) Hubbard site with `a†`, `a` (spinors) and `n` as single elements; that is what `hubbard_basis(U1Irrep, SU2Irrep)` should produce.
Energies were checked against the hand-built MPO of MPSKitExperimental to ~1e-13 on a random MPS.
Once step 5 lands, the same numbers should come out of the spin-free Hamiltonian of operator-bases.md §3.6 (`E[p, q] * E[r, s]`), with no Clebsch–Gordan code.

## 6. Where things are, and what bit us

* **The #38 material** lives in `/mnt/home/ldevos/Projects/MPSKitExperimental.jl` (Maarten's repository, PR maartenvd/MPSKitExperimental.jl#1, branch `lkdvos/fair-qchem-comparison_updated`).
  `examples/opsum_qchem_su2.jl` builds the terms; `examples/N2.STO3G.FCIDUMP` and `N2.CCPVDZ.FCIDUMP` are the integrals; `benchmark/qchem_comparison/` has the fused-MPO comparison and the DMRG timing scripts.
  That environment pins an older OpSum (`c69599b`, before `OperatorSum`) and needs Julia 1.11 or 1.12 (`ConcurrentCollections` does not build on 1.13).
* **Running the #38 terms against current OpSum** works from OpSum's own test environment (`jld --test`, which has MPSKit), by `include`-ing `opsum_qchem_su2.jl` and reading the FCIDUMP directly: a line `v i j k l` with `i, k > 0` is `(ij|kl)` (8-fold symmetric), with `k = l = 0` it is `h_ij`, and all zeros is `E₀`.
  That script folds an OpSum sign convention into its coefficients, `(-1)^{p(c)·p(in)}` per letter; anything that re-expresses its letters must carry that sign per letter, not per operator.
* **TensorKitTensors** is checked out at `/mnt/home/ldevos/Projects/TensorKitTensors.jl` (`e3be3c0`).
  With a symmetry it only provides neutral single-site operators and two-site products, which is why the charged elements come from factorisation.
* **The dense oracle is slow on the SU(2) Hubbard site**: `instantiate` compiled for ~90 s at `N = 4`; keep dense checks small there.
* **An oddity, triaged in step 6 (§13)**: lossless `SVDBondAlgorithm` gave 37 at the central bond of spinful all-to-all hopping where counting gives 22 (the vertex cover gives 42 today, 22 with the right basis).
  It is round-off: the lossless default `trunctol(rtol = eps)` keeps singular values made of the matrix-unit coefficients' round-off.
* **Conventions**: research notes are written one sentence per line; invariant checks are `throw`s through `_invariant`, never `@assert` (CLAUDE.md); format with Runic.

## 7. Step 1 as implemented (branch `feat/operator-basis`, for review)

Written without a go-ahead on the plan, at the author's request; everything below is open to change.

### Where the pass-through sentinel was used, and what changed

The sentinel `IrrepOperator(unit, 0)` became the letter `(unit, 1)`, the identity at element 1.
In the matrix-unit alphabet, which still exists, the trivial-charge matrix units move up by one (`n ≥ 2`); other charges are unchanged.

| file | use | change |
|---|---|---|
| `irreptensoroperators.jl` | `instantiate` of `n = 0` gave bare `id(V)`; `instances` gave `n ≥ 1` | `(unit, 1)` instantiates to `isomorphism(V ← V ⊗ C_unit)`, with the charge leg like every letter; trivial-charge matrix units at `n ≥ 2` (`_muoffset`) |
| `irrepkey.jl` | `passthrough`, `ispassthrough` | now `(unit, 1)` and `c == unit && n == 1` |
| `siteoperator.jl` | `SiteOperator(c)`, `isone` | unchanged code, through `passthrough`; docs |
| `localoperator.jl` | placement drops pass-through slots; `B + α` | unchanged; a projected identity factor is now *the* pass-through, so `project(X ⊗ 𝟙, B)[i]` places as a one-site term |
| `irrepalgebra.jl` | `O[i]` places the identity as a `K = 0` term; `_checklattice` letter ranges | placement unchanged; `_alphabet` counts the identity among the trivial-charge letters |
| `irreptermtable.jl`, `irrepgraph.jl`, `irrepgraph_svd.jl` | padding keys, idle sites, start/finish channel keys | unchanged: they build `passthrough(I)`, a letter no placed term contains, so the §2.2 start-channel invariant holds as before |
| `irrepmpo.jl` | `mpo_terms` skips pass-throughs; `_letter_optensor` special-cased the identity | walk unchanged; `_letter_optensor` is now just `instantiate` |
| `jordanmpo.jl`, `infinitegraph.jl`, `irrepgraph_vc.jl` | `isone` on bond entries | unchanged |
| `expterms.jl` | `nothing` in a string means pass-through | an identity letter in a string is mapped to `nothing`, so it is the same channel as the default string |
| `irrepinstantiate.jl` | — | `_space(site)` so the dense oracle accepts a `Basis` per site |

Behaviour change outside the plan: `instantiate(scalarop(c), V)` now has the trivial charge leg (`V ← V ⊗ C_unit`), so it adds to trivial-charge letters; before, it was `V ← V` and the sum threw.

### What step 1 adds

* `Basis(ops)` (`src/operators/algebra/basis.jl`): neutral single-site operators (or plain matrices without symmetry) are elements as given; neutral two-site operators are factorised into charged pieces (item 3 below); the identity is prepended; parallel elements of one charge are deduplicated; independence per charge is checked.
* `project(O, B)` for one site (a `SiteOperator`) and `project(h, B)` for `K` sites with the same basis in every slot (a `LocalOperator`), by changing coordinates from the exact matrix-unit expansion with the per-charge pseudo-inverse, then checking the residual like `project(h)` does.
* `instantiate(letter, B)` and `instantiate(::SiteOperator, B)`.
* The sweep is untouched; `test/test_operator_basis.jl` checks lossless MPOs and the dense operator read back through the basis for ℂ² (XX+YY+ZZ, bond 5 against 6 today, as in §5), U(1) spins, fermions (including a hop across a gap, projected as a 3-site block, against the gapped placement) and SU(2).

### Decided after review (2026-10-09)

1. **The letter index counts within a charge** (`(c, n)` is the `n`-th element of charge `c`), as for matrix units, not a global position in the basis.
   A `SymbolicMPO` can still hand out a coefficient vector in basis order through an accessor.
2. **A hand-made `Basis` takes neutral operators only.**
   `_basis_from_elements` (charged elements) stays internal; making it public later is additive.
   The other two unconfirmed items of §2 (`hop[i, i]`, model code without `couple`) are not touched by step 1.
3. **Factorised pieces are canonical, from a column-pivoted reduced row echelon form** of the pair's matrix-unit coefficients (`_factorise_pair`), not an SVD.
   Each piece's first coefficient is 1, and the pair's scale is its coefficient when projected into the basis.
   There is no rotation within a degenerate singular value, and a pair of single matrix units gives those matrix units: `XX + YY` gives `S⁺` and `S⁻`, pinned in `test/test_operator_basis.jl`.
   This closes the gauge item of §3.
4. **Parallel elements are dropped silently** for now, including a user's explicit identity; a non-parallel dependent element throws.
   An overcomplete basis may be wanted later; `project` would then need a rule for choosing among equivalent expansions.
5. **The matrix-unit alphabet and `project(O, V)` are removed in step 7**, with the builders.
6. ~~No basis on the lattice yet~~: done in step 3, below.

## 8. Step 3 as implemented

Done before step 2, since `instantiate(W)` needs the bases to ride along with the sum.

* `FiniteChain(bases)` / `FiniteChain(B, N)` and `InfiniteChain(bases)` / `InfiniteChain(B)` hold one `Basis` per site; mixed sites are `vcat(B₁, B₂, …)`, which is Base's `vcat` of non-array values and needed no code.
  The field is now `sites`, not `spaces`.
* A lattice is all bases or all spaces, of one symmetry (`_checksites`); a lattice of spaces still means the matrix-unit alphabet until step 7.
* Every consumer that needs the physical space reads `_space(site)`, and every one that interprets a letter calls `instantiate(letter, site)`: `_checklattice` (via `_alphabet(::Basis)`), `instantiate(H)`, `irrep_mpo_tensors` (finite and infinite) and `jordan_mpo_tensors`.
* `H'` projects each slot back into its site's basis (`project(h, Bs)`, one basis per slot, also public); a basis that is not closed under adjoints throws there, with the hint to extend it.
* Not done: `opsum(1:N)` without bases (§3.4), which needs `instantiate(W, bases)` from step 2.
* Tests: `test/test_basis_lattice.jl` (fermions with a hop across a gap against the matrix-unit oracle, densely, through `irrep_mpo_tensors` and in Jordan form; `H'`; a qubit–qutrit mixed chain; an infinite chain).

## 9. Step 2 as implemented

* `mpo(H)` returns a `SymbolicMPO` (`src/operators/compression/symbolicmpo.jl`): the lattice, per site a sparse matrix of `SiteOperator`s, per bond the charges and the indices of the two identity channels.
  On a finite chain it is computed like `jordan_mpo_tensors` (`_irrep_channels`, finish channel forced into the cover); with decaying channels it has no Jordan form yet and says so.
* `instantiate(entry, B)` is the step-1 `instantiate(::SiteOperator, ::Basis)`.
* `instantiate(W)` gives MPSKit's `FiniteMPOHamiltonian` or `InfiniteMPOHamiltonian`, from `ext/OpSumMPSKitExt.jl` (MPSKit is a weak dependency, compat `0.14`).
  The infinite Jordan form is new: every bond of the cell carries both identity channels, so each bond is reordered by its own `start`/`done` and nothing is padded.
* `examples/mpskit.jl` and `test/test_jordan_mpo.jl` use `instantiate(mpo(H))` instead of the hand-written `to_jordan`; the example now also runs VUMPS on an infinite SU(2) chain (−0.443136 against Bethe's −0.443147).
* `test/test_mpskit_ext.jl` compares `instantiate(W)` with MPSKit's own construction from TensorKitTensors operators on random states (fermions with a hop across a gap, finite and infinite; U(1)×U(1) and SU(2) Hubbard), and pins three rows of §5: Hubbard U(1)×U(1) `N = 10` 10 → 6, SU(2) Hubbard `N = 4` 10 → 6, spinful all-to-all hopping 42 → 22.
* `research/n2_basis_acceptance.jl` (run with `jld --test run`) reproduces the N₂ rows of operator-bases.md §1 exactly: STO-3G 347 / 677 → 102 / 192 with 860 term tuples instead of 7,295; cc-pVDZ bonds 7, 14, 21: 737 / 1451 → 212 / 408, 2984 / 5924 → 786 / 1542, 765 / 1507 → 226 / 436.
  The basis is the complete SU(2) Hubbard site (`𝟙, n, nꜛꜜ, a†, n·a†, a, n·a, S⃗, Δ†, Δ`), built from neutral pairs written with the #38 script's own machinery, and the terms are re-expressed letter by letter (`OpSum._rebasis`).
  The two MPSKit Hamiltonians agree to ~1e-14 on a random 14-electron MPS; this is a check of the basis route against the matrix-unit route, both through the new extension, not against MPSKitExperimental's hand-built MPO.
* `Basis` now chops pseudo-inverse round-off (entries below `rtol` of the largest), which would otherwise become letters with ~1e-16 coefficients.
  Re-expressing whole Hamiltonians also leaves ~1e-15 terms where different matrix-unit tuples cancel on one basis term; the acceptance script drops those, and a public route for re-expressing terms (step 7) will need the same.
* `irrep_mpo`, `FiniteMPO` and `InfiniteMPO` are removed in a separate change (§10).
  `instantiate(W, bases)` for `opsum(1:N)` (§3.4) is not done.
* The test environment's manifest needed a full `Pkg.update()` (MPSKit `main` is 0.14 now; `test/Project.toml` already asked for it), so BlockTensorKit 0.3.20 and TensorKit 0.17.2 locally.

## 10. `irrep_mpo` → `mpo`

* `irrep_mpo`, `FiniteMPO` and `InfiniteMPO` are gone; `mpo(H)` returns a `SymbolicMPO` on either lattice, so `using OpSum, MPSKit` no longer leaves anything ambiguous.
  A `SymbolicMPO` iterates over its sites and does not destructure into `(Ws, bondsectors)`; tests use `reduced(W)` from `test/testutils.jl`, other code the fields `W.Ws`, `W.bondsectors`.
* `irrep_mpo_tensors(W)`, `jordan_mpo_tensors(W)`, `mpo_terms(W)`, `mpo_terms_window(W, ncells)` take the `SymbolicMPO` alone, since it carries its lattice; the identity channels are `W.start` / `W.finish` (`done` before).
* One behavioural difference: `mpo` on a finite chain without channels forces the finish channel into the cover (what the Jordan form needs), where `irrep_mpo` did not.
  On every showcase model and on the N₂ acceptance numbers that costs nothing, but a constant (`K = 0`) term next to two two-site terms costs one trivial-charge index at the first bond; `test/test_irrep_graph.jl` pins both covers.
* `jordan_mpo_tensors` (and so `instantiate`) now also covers infinite chains with decaying channels, checked against MPSKit with the tail truncated at range 40 in `test/test_mpskit_ext.jl`; a finite chain with decaying channels still has no Jordan form.

## 11. Step 4 as implemented

Confirmed by the author before starting (2026-10-11): `hop[i, i]` is an on-site product (the second unconfirmed item of §2), and only products that share a site are resolved densely in this step; interleaved non-abelian `*` without a shared site stays for step 5's F/R moves.

* `a * b` on placed operators (`src/operators/algebra/products.jl`) is the operator product in written order.
  Every product is written as one caterpillar in written order, coefficient `cₐ c_b (-1)^{p(a)p(b)}` with `p` the fermion parity of each factor's total — the graded sign `couple` applies; with it `c†ᵢ * cᵢ = nᵢ` and `cᵢ * c†ᵢ = 1 - nᵢ` hold.
* On distinct sites the product is a `Term` at once: as written when in site order; reordered with one R-symbol per swapped pair under an abelian symmetry (what `couple` does when it inserts a leg); and under a non-abelian one, neutral factors on disjoint, non-interleaved supports are swapped (they commute), interleaved ones throw (step 5).
  `*` needs a forced channel: always under an abelian symmetry, neutral factors under a non-abelian one.
* A product that repeats a site is a `PendingTerm` in the bag's new `pending` vector; such a bag has no normal form, so `length`, iteration and `≈` throw (and say why), `show` reports it, and `couple`, `dot` and `expterm` refuse it.
  `opsum!` resolves it with the lattice: the letters' embedded tensors multiplied in written order, their charge legs fused by the written caterpillar (`_written_block`), projected back into the sites' bases (or matrix units) and placed, memoised per shape. A product outside a basis' span throws there with the hint to extend it — e.g. `a·n` on the SU(2) Hubbard site built from `e⁺e⁻`, `e⁻e⁺`, `n` alone.
* Placement on sites that are not strictly increasing is the same product: `hop[i, i]` an on-site product, and `hop[i + 1, i]` (abelian) the reordered operator, so the hermitian-conjugate partner of a pair no longer needs a second pair.
* `test/test_products.jl` checks everything against the dense operator product: single-site fermion identities, products of hopping operators sharing sites, `hop[i, i]` and reversed placement in a basis, SU(2) `S⃗·S⃗` on one site (`3/4`), products of exchange pairs sharing a site or disjoint, the SU(2) Hubbard `Σσ a†σ aσ` on one site (`n`), and an infinite chain.
* Not done: the spin-free chemistry Hamiltonian of operator-bases.md §3.6 needs interleaved non-abelian products (step 5); its coincident-index terms already work.

## 12. Step 5 as implemented

* **Re-coupling** (`src/operators/algebra/products.jl`). A caterpillar is a charge-only tensor `C₁ ⊗ … ⊗ C_K ← C_tot` with one nonzero entry at its tree (`_treetensor`) — the tensor `_instantiate_term` fuses the letters' charge legs with — so rewriting that tensor rewrites the term.
  `_recouple` permutes its legs into site order with TensorKit's `permute` and reads the result back in the caterpillar basis (`_expand`); `_concat` fuses two caterpillars into one when the first is charged under a non-abelian symmetry (the F-move of §3.6). Both are memoised per shape (`_RECOUPLE_CACHE`), so no 6j algebra is written here.
  The single-tree `braid`/`permute` of the TensorKit in use asserts `UniqueFusion`, and its pair/block form needs `bendleft`, which does too; the dense charge tensor sidesteps both and is tiny (multiplicity 1 per leg).
  Gated on `_canrecouple` (multiplicity-free, symmetric braiding); the abelian fast path (one R-symbol per swapped pair) is unchanged, and `test/test_products.jl` checks the two agree on fermions.
* **Where it is used.** `couple` with an out-of-order operand under a non-abelian symmetry (one term can become several), `*` (now any pair of factors whose totals fuse to a single sector, not only neutral ones), and placement on out-of-order sites (`B[2, 1]`, `project(h, [2, 1])`). The three tests that pinned the old throws now check the result.
* **Oracle.** `_written_block` materialises a written-order caterpillar as written; the tests compare it with the re-coupled site-order terms (spin-1 on three and four sites, every channel), plus `E[q, p] = E[p, q]†` and interleaved products of spin-summed hoppings on the SU(2) Hubbard site against the dense product.
* **Acceptance** (`research/n2_spinfree_acceptance.jl`): the spin-free Hamiltonian `Σ h E + ½ Σ (pq|rs)(E E - δ E)` of §3.6.
  * Against an exact Jordan–Wigner Fock-space spectrum (plain matrices, no TensorKit) on 4–5-orbital sub-blocks of both FCIDUMPs: agreement to ~1e-12.
  * Against `opsum_qchem_su2.jl` re-expressed in the basis, full size: STO-3G the same 860 terms and the same 102 / 192 at the centre bond; cc-pVDZ the same bond dimensions at bonds 7, 14, 21 (212 / 408, 786 / 1542, 226 / 436). Built in 3.6 s for cc-pVDZ once warm.
  * **Finding:** `opsum_qchem_su2.jl` is *not* a reference for energies. Under the current OpSum conventions its two-site strings carry the opposite sign of the physical `a†_p a_q`; its own comment describes a per-letter sign fix written against OpSum before the graded sign of #36. It misses the exact spectrum on a generic one-body `h` (by 0.65 on a 3-orbital example), and on orbitals 1:5 of STO-3G by 1.5e-2; it happens to be exact on some sub-blocks, which is how it survived. TensorKitTensors' Hubbard operators and the spin-free route agree with the exact spectrum. So the energies of §5 / step 2 compared two copies of that Hamiltonian with each other; the bond dimensions are unaffected (a sign does not change the term set).
* The spin-free cc-pVDZ sum has 216 more terms than the qchem one (37 616 vs 37 400), none missing: all are cancellations left at round-off (at most 1.5e-17 of the largest coefficient), which the qchem path filters with a relative cut and `canonicalize!` does not. They cost no bond dimension.

## 13. Step 6 as implemented

* **Truncation in the basis' metric** (`src/operators/compression/metric.jl`). The SVD backends treat a site's letters as orthonormal, which a `Basis` need not be (`n` overlaps `𝟙`).
  The Hilbert–Schmidt norm of a caterpillar term factorises, `‖t‖² = |coeff|² Πᵢ dot(aᵢ, aᵢ)/dim(cᵢ)` times `dot(𝟙, 𝟙)` per idle site (checked numerically on the SU(2) Hubbard site, and pinned in `test/test_basis_svd.jl`), so the metric is a Gram matrix per charge, `G_c[n, m] = dot(eₙ, eₘ) / (dim(c) dot(𝟙, 𝟙))`, in which the identity has norm 1.
  A truncating `SVDBondAlgorithm` on a lattice of bases rewrites the terms over orthonormal letters (`G = A'A`, Cholesky, so a letter picks up components of the earlier letters of its charge, including the identity, which is then dropped like a pass-through), compresses, and maps every site entry back with `A⁻¹`.
  `A` is upper triangular with `A[1, 1] = 1`, so the identity, the pass-throughs and the identity channels are untouched.
  Lossless SVD (`trunc === nothing`) skips it: the rank does not depend on the basis, and the rewrite would only add round-off.
  Sites given by spaces (matrix units) keep the plain metric, since that alphabet is not a basis.
  Effect: the same Ising-type Hamiltonian in `{X, Y, Z}` and in `{X, X + Y, Z + 0.7𝟙}`, truncated to rank 2–4, now gives the same operator (to 1e-15); before, the overlapping basis truncated to relative errors 0.97 / 0.25 / 0.12 against 0.69 / 0.14 / 0.055.
* **The 37-vs-22 oddity of §6 is round-off.**
  In matrix units the lossless default `rtol = eps` gives 37, `rtol = 1e-15` gives 27 and `1e-14` gives 22; in the basis, where the coefficients are exact, both SVD sweeps give 22 by default.
  Not changed: raising the lossless default tolerance is a decision for the author.
* **Decaying channels over bases.** The string is an operator in the sites' basis like any other (`string = project(P, B)`); `test/test_basis_lattice.jl` checks a decaying fermion hop with a parity string against the explicit sum of products, densely on a finite chain and term by term on an infinite one.
  New check on insertion (`_checkchannels`): a channel's letters land on sites it does not name (its translates move by one site on a finite chain; a string fills every site of the gap), so those sites must carry one basis, and the letters must exist in it.
  Before, a finite chain did not check channel letters at all, and string letters were never checked.
* SVD on an infinite chain or with decaying channels still throws, as before (both need the whole bond matrix up front).

## 14. Step 7 as implemented

* **Stock bases** (`src/operators/algebra/stockbases.jl`): `spin_basis`, `fermion_basis`, `boson_basis`, `hubbard_basis`, `tj_basis`, memoised functions of the symmetry on TensorKitTensors' spaces.
  Each is defined from an ordered list of TensorKitTensors operators; one the symmetry does not admit (TensorKitTensors throws) or whose pieces are already spanned is skipped (`_basis_from_elements(…; skipdependent = true)`), so one list serves every symmetry, and the basis is completed with the matrix units it lacks, so products and `H'` never leave it.
  The order is what makes the common operators atomic: `S⃗·S⃗` under SU(2), `e⁺e⁻` on the U(1)×SU(2) Hubbard site (the #38 site, 10 elements), `e_hopping` under an SU(2) particle symmetry, `Sᶻ` and the `S±` pairs under U(1), `SᶻSᶻ` under ℤ₂ — pinned in `test/test_stock_bases.jl`.
  `Basis(ops; complete = true)` completes a hand-made basis the same way. TensorKitTensors is a full dependency.
* **Builders**: `spin_ops(sym; spin)`, `fermion_ops(sym)`, `boson_ops`, `hubbard_ops`, `tj_ops` return every neutral TensorKitTensors operator of the site projected into the stock basis, **named after the TensorKitTensors function** (`spin_ops(SU2Irrep).S_exchange[i, j]`, `fermion_ops().f_plus_f_min[i, j]`, `hubbard_ops(U1Irrep, SU2Irrep).e_plus_e_min`). Operators the symmetry does not admit as neutral ones are absent. The builders take the symmetry, not a basis, since a `Basis` does not record which site it came from (§3.7's `fermion_ops(B)` sketch). Field names were not fixed by the design; this choice is open to review.
* **Removed**: `spin`, `matrixunit`, the space-based `spin_ops`/`fermion_ops`, `project(h)`, `project(h, sites)`, `project(O, V)`, and lattices of spaces (`FiniteChain(V, N)`; `opsum` with spaces throws an `ArgumentError` too). The matrix-unit expansion stays internal (`_project_units`, `instances(IrrepOperator, V)`, `instantiate(letter, V)`): it is the coordinate system `Basis` is defined and projected through. A charged single-site tensor still projects onto a basis' charged elements (`project(t, B)`), which is how `couple` and `dot` get operands; both stay exported.
* **Migration** (one sonnet agent per group of files, against a fixed contract): tests that need charged letters use `test/testutils.jl`'s `unitbasis(V)` (identity plus every matrix unit but the first trivial one), `spinvector`, `matrixunit`, `spinladder`, `fermionmode`; the rest use stock bases.
  Every pinned bond dimension of the sweep, infinite-chain and decaying-channel tests is unchanged. Tests deleted: only those whose subject was a removed function (builder argument validation, `matrixunit` memoisation, `project(O, V)` forms).
  One pin lost its meaning: the "matrix units" side of the XX+YY+ZZ comparison (`test/test_operator_basis.jl`) gives 5, not 6, because `unitbasis` gives `Z` an identity component where the old alphabet had two projectors. The Hubbard 10 → 6 and all-to-all 42 → 22 comparisons are unchanged.
* **Examples and docs** use the stock bases, builders, `*` and placement; `couple` appears only in one "charged building blocks" section of `docs/src/operators.md`. Numbers on the pages that dropped: TFIM `N = 6` 4 → 3, XXZ U(1) 6 → 5, AKLT / bilinear-biquadratic 5 → 4, SU(2) Hubbard 6 → 4 (dense 10 → 6, the §5 acceptance number). The ladder-with-plaquettes example now writes the physical ring exchange (products of exchange pairs) instead of one `j₁₂ = 0` channel, a different operator, at 7 / 17. `examples/common.jl` gained `exchange(Ba, Bb)`, the S⃗·S⃗ of two different spins as a projected `TensorMap` (mixed-spin chains have no stock builder).
* **Showcase models** (`benchmark/ShowcaseModels.jl`) are built from the stock builders; `hubbard_su2` drops from 10 / 6 to 6 / 4 (dense / reduced), every other model is unchanged at the CI sizes.
* **Acceptance** (`research/n2_*.jl`): the matrix-unit side is now the bare sweep over the #38 terms (347 / 677 at the STO-3G centre bond, unchanged), the basis side 102 / 192; the spin-free route still matches the exact spectrum (~6e-13), and so does `hubbard_ops(U1Irrep, SU2Irrep).e_plus_e_min` in the stock basis (3.8e-13). The basis script no longer compares MPSKit energies (§12: that compared one wrong-sign Hamiltonian with itself).
* **Noticed, not changed**: `≈` on term sets is exact in the term set, so a round-off leftover (`ex[1,2]*ex[2,3] - ex[2,3]*ex[1,2]` leaves a 2e-16 term) breaks it; letter validation only checks that a letter exists, so spin-1 `S_exchange` placed on a spin-½ SU(2) site is not rejected (decision 1 of §2).
* Still open: `opsum(1:N)` with `instantiate(W, bases)` (§3.4), an overcomplete basis (§7, item 4), the rest of MPSKitModels (lattices, named models; §3).
