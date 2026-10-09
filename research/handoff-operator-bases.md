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
* **An untriaged oddity**: lossless `SVDBondAlgorithm` gave 37 at the central bond of spinful all-to-all hopping where counting gives 22 (the vertex cover gives 42 today, 22 with the right basis).
  It looks like an SVD-backend issue unrelated to this work.
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
