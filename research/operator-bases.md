# On-site operator bases: an explicit `Basis` at both ends of the pipeline

Prompted by [#38](https://github.com/lkdvos/OpSum.jl/issues/38): an SU(2) quantum-chemistry MPO whose bond dimension is about 3× that of a hand-built complementary-operator MPO.
This note records the approach agreed after the design meeting, the evidence behind it, and what is still open.

**Summary.**
The pipeline currently fixes the on-site basis implicitly: every operator is split into matrix units before the compression sees it, and an operator that is a sum of matrix units on both sides of a bond cannot be shared.
The agreed fix is to make the basis an explicit object.
The user builds a `Basis` from operators of their choice, projects their operators into it, writes the Hamiltonian with the resulting symbolic operators, and instantiates the symbolic MPO with the same basis.
The compression in between only ever sees basis indices, so whatever the user puts in the basis is never split.

---

## 1. The problem, with numbers

`couple` distributes composite operands over their matrix-unit letters:

```julia
cdu = matrixunit(Vh, u, e) + matrixunit(Vh, ud, d)    # c†↑ : two matrix units, one charge
couple(cdu[i], cu[j])                                  # → 4 terms, one per letter pair
```

At bond `i|j` the sweep sees two prefixes and two suffix classes, all connected, and the vertex cover spends 2 indices where the exact rank is 1.
The cover is optimal for the sparsity pattern it is given; the pattern is wrong because of the basis.
Every channel whose operator is a `k`-letter sum costs `k` instead of 1.

Measured on `main` (`5f246be`), `sum(dim, bondsectors)`, against the minimum (lossless SVD or counting):

| model | site | today | minimal |
|---|---|---|---|
| `XX+YY+ZZ` chain, `N = 8` | `ℂ²` | `5, 6, …, 6, 5` | `4, 5, …, 5, 4` |
| `(X+Y)⊗(X+Y)`, `N = 8` | `ℂ²` | `3, 4, …, 4, 3` | `2, 3, …, 3, 2` |
| spinful NN hopping, `N = 10` | `U₁⊠U₁` | `9, 10, …, 10, 9` | `5, 6, …, 6, 5` |
| spinful all-to-all hopping, `N = 10`, centre | `U₁⊠U₁` | 42 | 22 |

On the #38 SU(2) site `(0,0,0) ⊕ (1,½,1) ⊕ (2,0,0)` both spinors `a†` and `a`, and `n`, are two matrix units each.
Re-running the #38 term construction (`MPSKitExperimental.jl/examples/opsum_qchem_su2.jl`) with a user-chosen basis in which they are single elements, with the sweep unchanged, gives (channels / total dimension):

| N₂ | bond | hand-built (fused) | OpSum today | OpSum, user basis |
|---|---|---|---|---|
| STO-3G | 5 (centre) | 61 / 238 | 347 / 677 | **102 / 192** |
| cc-pVDZ | 7 | 128 / 478 | 737 / 1451 | **212 / 408** |
| cc-pVDZ | 14 (centre) | 421 / 1678 | 2984 / 5924 | **786 / 1542** |
| cc-pVDZ | 21 | 135 / 478 | 765 / 1507 | **226 / 436** |

`⟨H − E₀⟩` on a random MPS agrees with the hand-built MPO to ~1e-13.
The total bond dimension falls about 3.8× and ends up 8–20% *below* the hand-built MPO's.
Term construction also gets faster: 860 term tuples instead of 7,295 for STO-3G, built in 1.3 s instead of 28 s.
So a user-chosen basis is enough to close the gap in #38; the SU(2) coupling paths suspected in the issue are not a significant factor.

---

## 2. Why this is cheap to support

> **Changing the on-site basis is a linear map per site and per charge, on that charge's multiplicity space.**
> **It commutes with the caterpillar fusion tree, so it never touches F-, R- or B-symbols.**

All operators of charge `c` on `V` are maps `V ← V ⊗ C_c`, and the caterpillar embedding is linear in each factor.
Any basis of those maps can therefore stand in for the matrix units, with the *same* running bond charges and vertex labels.
The term algebra and the sweep only compare letters, and the fermionic signs and `dot` factors depend only on charges, so none of them changes.
What changes is the meaning of a letter, which matters at exactly two places: where operators enter (`project`) and where tensors leave (`instantiate`).
That is where the agreed design puts the basis.

---

## 3. The design

### 3.1 Decisions

1. **A `Basis` is an ordered list of operators on one space `V`.**
   Internally each element is a symmetric `TensorMap` `V ← V ⊗ C_c` of one definite charge `c`, and the basis groups its elements by charge; plain matrices are the no-symmetry case.
   Users never handle the charged elements: a basis is *defined* from neutral operators, and two-site ones are factorised into their single-site charged pieces (§3.7).
2. **The identity is an ordinary element, by convention always element 1** of each space's basis.
   It replaces today's pass-through sentinel: idle sites and the Jordan identity corners are expressed in it.
3. **Operators are projected into a basis**: `project(O, basis)` gives a *symbolic* operator, a coefficient vector over the basis elements.
   The operators a user projects and writes with are always neutral: single-site neutral operators, and neutral pairs such as hopping or exchange.
4. **Each site has a basis, and spaces come from the bases.**
   A sum is built over a list of per-site bases, wrapped as `FiniteChain(bases)` or `InfiniteChain(bases)` (a unit cell); a bare vector or `1:N` means a finite chain.
   There is no separate lattice of spaces.
5. **Several spaces on one lattice** (e.g. fermions and spins on different sites) are combined with `vcat(basis_f, basis_s)`.
   Indices stay local to each site's basis: a letter is `(charge, index)` as today, and index 1 is the identity of whichever basis the site has.
6. **Letters carry no basis tag.**
   A symbolic operator placed on a site with a different basis than the one it was projected into is reinterpreted silently; that is accepted for now.
7. **`*` multiplies operators, and is defined whenever the overall channel is forced.**
   That is always the case under an abelian symmetry, and under any symmetry when the factors are neutral.
   Since every user-facing operator is neutral (decision 3), every product in model code has a forced channel; `*` only throws on charged operands whose channel is not forced, which model code never produces.
   On the same site, `a[i] * b[i]` is the on-site product: it is kept unevaluated and resolved in the site's basis when the term enters the sum (§3.5).
8. **The MPO is symbolic**: `mpo(H)` returns a `SymbolicMPO` whose entries are coefficient vectors over the basis, and `instantiate(entry, basis) = Σ coeff · element` turns them into tensors.
   `instantiate(W)` of a whole MPO returns MPSKit's own struct (`FiniteMPOHamiltonian` or `InfiniteMPOHamiltonian`), through a package extension.
   The name `SymbolicMPO` avoids `FiniteMPO`/`InfiniteMPO`, which MPSKit also exports.
9. **`project` always takes a basis.**
   The basis-free forms (`project(O, V)`, `project(h)` into matrix units) go away for now.
10. **No automatic basis discovery.**
   The basis is always the user's choice; what is in it is atomic, and what is not gets expressed in it.
11. **`couple` and `*` accept any site order, under any symmetry.**
   Non-abelian factors that arrive out of site order are re-coupled into the site-ordered caterpillar with F- and R-moves (§3.6).
   This lifts today's rule that non-abelian `couple` must be written in site order.

### 3.2 The worked example

```julia
# ---- a basis: the identity first, then whatever the user wants atomic
Imat = [1 0; 0 1]; Xmat = [0 1; 1 0]; Ymat = [0 -im; im 0]; Zmat = [1 0; 0 -1]
basis = Basis([Imat, Xmat, Ymat, Zmat])

# ---- project operators into it
Splus, Smin = [0 1; 0 0], [0 0; 1 0]
Splus_symbolic = project(Splus, basis)      # [0, 1/2,  im/2, 0]: two elements, so S⁺S⁻ costs 2 channels
Smin_symbolic  = project(Smin, basis)       # [0, 1/2, -im/2, 0]

# the same operators in a basis where they are atomic
alternativebasis = Basis([Imat, Splus, Smin, Zmat])
Splus_symbolic = project(Splus, alternativebasis)   # [0, 1, 0, 0]
Smin_symbolic  = project(Smin, alternativebasis)    # [0, 0, 1, 0]

# ---- the sum: `*` on different sites couples to the unit charge
H = opsum(1:(N - 1)) do i
    return Splus_symbolic[i] * Smin_symbolic[i + 1]
end

# ---- the symbolic MPO, and its instantiation
W = mpo(H)                                         # a SymbolicMPO (not `mpo = mpo(H)`, which shadows the function)
for o in W, (i, j) in eachindex(o)
    tensor_dense = instantiate(o[i, j], alternativebasis)   # Σ coeff .* basis
end
Hmpskit = instantiate(W, fill(alternativebasis, N))       # an MPSKit.FiniteMPOHamiltonian
```

### 3.3 Symmetric sites

With symmetry the elements are charged `TensorMap`s and the basis is block-diagonal by charge, but the user only ever names neutral operators:

```julia
P, S = U1Irrep, SU2Irrep
basis = Basis([e_plus_e_min(P, S), e_min_e_plus(P, S), e_num(P, S), ud_num(P, S), …])
hop = project(e_plus_e_min(P, S), basis)     # a symbolic neutral pair: one coefficient on (c†) ⊗ (c)
n   = project(e_num(P, S), basis)            # a single-site neutral operator
H = opsum(FiniteChain(fill(basis, N))) do i
    -t * (hop[i, i + 1] + hop[i + 1, i]) + U * project(ud_num(P, S), basis)[i]
end
```

* `Basis` factorises the two-site operators into their charged single-site pieces (here the spinors `c†` and `c`), and uses the single-site ones as they are; element 1 is always the identity, of the trivial charge.
* A two-site symbolic operator is placed on two sites, `hop[i, j]`; placing it on one site, `hop[i, i]`, is an on-site product, resolved on insertion like `a[i] * b[i]` (§3.5).
* Internally, an element must have one charge, and an operator mixing charges is projected onto elements of several charges.
* The elements of one charge must be linearly independent.
  They need not be orthonormal, so `project` solves against the basis per charge rather than taking inner products.

### 3.4 Building the sum

```julia
H = opsum(FiniteChain(bases)) do i   # bases[i] is the basis of site i; fill(basis, N) for a uniform chain
    …
end
H = opsum(InfiniteChain(cell)) do i … end   # a unit cell of bases; terms may reach past the cell
H = opsum(1:N) do i … end            # only the site range; the bases are supplied at `instantiate`
```

* The bases given to `opsum` ride along to the MPO, so `instantiate(W)` needs no further arguments; with `1:N` they are passed to `instantiate` instead.
* There is no check that an operator was projected into its site's basis (decision 6).
  The one cheap check left is that each letter `(c, k)` exists in the site's basis.
* `a[i] * b[j]` with `i ≠ j` is `couple(a[i], b[j]; to = c)` with the forced channel `c`.
  The order is free under any symmetry: the braiding phase (the fermionic sign) is inserted as today, and non-abelian factors are re-coupled (§3.6).
  `couple(…; to = c)` stays the underlying primitive, which `*` calls with the forced channel; model code built from neutral operators does not need it.
* `a[i] * b[i]` is the on-site product, described in §3.5.
* `dot`, multi-site `project(h, bases)` blocks, `expterm` and `H'` keep their current meaning; `project` of a `K`-site block expresses each slot in that slot's basis.
  `H'` needs the bases, so it is available on sums built over bases, not on `opsum(1:N)`.

### 3.5 On-site products, resolved on insertion

A symbolic operator does not know its basis (decision 6), so `a[i] * b[i]` cannot be evaluated when it is written.
The site's basis *is* known when the term enters the sum, so the product is kept symbolic until then:

```julia
H = opsum(FiniteChain(bases)) do i
    return n[i] * n[i] + hop[i, i + 1] * n[i]          # products on site i, in the order written
end
```

* **The node.**
  `a[i] * b[i]` (and longer products such as `a[i] * b[j] * c[i]`) puts an unevaluated product node at site `i`: the letters in the order they were written.
* **Charges and signs.**
  Under an abelian symmetry the node's charge is forced by its factors, so running bond charges and braiding phases are computed from charges alone, as today.
  Bringing `c[i]` next to `a[i]` past `b[j]` is the same R-symbol bookkeeping `couple` already does when it reorders legs.
  Under a non-abelian symmetry the same move is an R-move, and fusing the node's adjacent legs into one leg is an F-move (§3.6).
* **Resolution.**
  `opsum!` evaluates every node in its site's basis — instantiate the factors, multiply, project back — memoised per (basis, letters).
  The resolved terms then enter the normal form like any other; a product outside the span of the basis throws there.
* **What it requires.**
  * The bases: `opsum(1:N)` cannot hold products, since there is nothing to resolve them in and the sweep needs letters.
    It throws and asks for bases, as `H'` does.
  * A bag with unresolved products is not canonical, since such terms cannot be merged with others.
    `length`, `≈` and `show` are therefore only meaningful on the sum, after insertion; on a bare `Terms` bag they throw while products are pending.
  * Until insertion a term has two kinds of site factor: a resolved key, or a product node.
    The sweep only ever sees resolved terms.
* **Non-abelian symmetry.**
  Two charged operators on one site can fuse to several charges `e`.
  Only the overall channel of the term is forced; the node's own charge is not, so the F-move that fuses its legs produces one term per `e`, each with its F-symbol coefficient.
  Each of those is resolved in the basis with the fusion tensor of its channel, `C_a ⊗ C_b → C_e`, memoised per (basis, letters, channel).

Since products are resolved through the site, `SiteOperator` needs no reference to its basis: it is a plain coefficient vector.

### 3.6 Re-coupling under a non-abelian symmetry

A term is stored in one normal form: factors in site order, coupled as a left-nested caterpillar, with the running bond charges as the tree.
Under an abelian symmetry any product is brought into that form by reordering legs and collecting R-symbols, which `couple` already does.
Under a non-abelian symmetry the product of two caterpillars whose sites interleave is a *linear combination* of caterpillars, with coefficients from F- and R-symbols.

* **The operation.**
  `couple`/`*` merges the two operands' trees, braids the legs into site order, and recouples into the caterpillar basis.
  TensorKit provides each step on fusion trees (`merge`, `braid`/`permute`, `Fsymbol`), so OpSum calls them rather than implementing 6j algebra.
* **Same-site legs.**
  After braiding, legs on the same site are adjacent; the F-move that fuses them into one leg of charge `e` gives the product nodes of §3.5.
* **Cost.**
  The coefficients depend only on the *shape* of a product — the charges, their positions and which legs share a site — so they are cached per shape.
  This is the pattern cache of the #38 script, computed by TensorKit instead of by hand, and its size does not grow with the number of sites.
* **Behaviour change.**
  Non-abelian `couple` with operands out of site order currently throws; under this design it succeeds.
  The tests that pin the throw become tests of the re-coupled result against the dense oracle.

The payoff is that spin-adapted models need no channel at all.
With the spin-summed excitation operator `E_pq = Σ_σ a†_pσ a_qσ`, a neutral pair, and `Σ_στ a†_pσ a†_rτ a_sτ a_qσ = E_pq E_rs − δ_qr E_ps`:

```julia
B = hubbard_basis(U1Irrep, SU2Irrep)
E = project(e_plus_e_min(U1Irrep, SU2Irrep), B)       # E[p, q] = Σσ a†_pσ a_qσ; E[p, p] = n_p
H = opsum(FiniteChain(fill(B, N))) do …
    h[p, q] * E[p, q] + ½ * eri[p, q, r, s] * (E[p, q] * E[r, s] - δ(q, r) * E[p, s])
end
```

Every `*` here multiplies neutral pieces, so every channel is forced.
Coinciding indices are on-site products resolved on insertion, so they need no special case.
This replaces the hand-written Clebsch–Gordan code of `opsum_qchem_su2.jl`.

### 3.7 Stock bases and the operator builders

The builders take a basis and return the neutral operators of that site as symbolic operators:

```julia
B = hubbard_basis(U1Irrep, SU2Irrep)          # memoised: repeated calls return the same object
ops = fermion_ops(B)                          # (; n, nud, hop = e_plus_e_min, hop′ = e_min_e_plus, …)
ex  = spin_ops(spin_basis(SU2Irrep)).SS       # the exchange pair S⃗ᵢ·S⃗ⱼ, one element on each site
X   = project(pauli_x(), spin_basis(Trivial)) # any neutral TensorKitTensors operator, projected
```

* Each builder is `project(tensor, basis)` of the corresponding TensorKitTensors tensor, so its result is atomic exactly when the basis makes it so; `project` accepts any neutral single-site or two-site TensorKitTensors operator.
* A stock basis knows its own layout, so `spin_ops` no longer needs the list of sectors that fixes the `m` order today.
* The stock bases are functions of the symmetry, named after the TensorKitTensors space they live on, and their spaces *are* those spaces, so state conventions match TensorKitTensors by construction:

| basis | space | defined from the neutral operators | resulting charged elements |
|---|---|---|---|
| `spin_basis(symmetry; spin = 1//2)` | `spin_space` | `S⁺S⁻`, `S⁻S⁺`, `Sᶻ` (no symmetry, U(1)); `SS` (SU(2)) | `S⁺, S⁻` or `S⃗`; higher multipoles for `spin > ½` |
| `fermion_basis(symmetry)` | `fermion_space` | `f⁺f⁻`, `f⁻f⁺`, `n` | `c†, c` |
| `hubbard_basis(particle, spin)` | `hubbard_space` | `e⁺e⁻`, `e⁻e⁺`, `n`, `nꜛꜜ`, `singlet⁺`, `singlet⁻`, `S⁺S⁻`/`SS` | `c†σ, cσ` (spinors under SU(2)), `Δ†, Δ`, `S±` or `S⃗` |
| `tj_basis(particle, spin)` | `tj_space` | the t-J counterparts | the projected subset of the Hubbard elements |
| `boson_basis(symmetry; cutoff)` | `boson_space` | `b⁺b⁻`, `b⁻b⁺`, `n` | `b†, b` |

Every stock basis is completed so that it spans all operators on its space, so that on-site products and `H'` never leave it.
The tensors come from TensorKitTensors.jl; together with this package that is meant to replace MPSKitModels.jl entirely.

**How a basis is built from neutral pairs.**

* `project` of a neutral two-site operator into matrix units gives, per charge `c` of the first site, a coefficient matrix over the letters of `c` on the first site and of its dual on the second.
* Its rank factorisation gives the charged single-site pieces: for `e⁺e⁻` under SU(2), rank 1 in the spinor channel, with `c†` on the left and `c` on the right.
* Pieces from all defining operators are collected, deduplicated by direction, checked for independence, put after the identity and the single-site neutral operators, and completed to span the space.
* Each piece's normalisation and phase are a gauge choice that the user never sees: user-facing operators are neutral pairs, so a rescaled factor only rescales the pair's coefficient.
  The convention is canonical: the rank factorisation is a column-pivoted reduced row echelon form, not an SVD, and each piece's first coefficient is 1.
  So there is no rotation within a degenerate singular value, and a pair of single matrix units gives exactly those matrix units (`XX + YY` gives `S⁺` and `S⁻`).

### 3.8 Forming and instantiating the MPO

* `mpo(H)` runs the existing sweep on basis indices and returns the symbolic MPO: per site a sparse matrix of coefficient vectors, per bond the charges.
* `instantiate(entry, basis)` gives one entry's tensor.
* `instantiate(W[, bases])` builds MPSKit's `FiniteMPOHamiltonian` or `InfiniteMPOHamiltonian` directly, in a package extension that loads with MPSKit.
  It loops over the entries in Jordan form, with element 1 at unit coefficient emitted as a `BraidingTensor`, and replaces the hand-written `to_jordan` adapter.
* Verification stays available: `instantiate(H)` on a finite chain, and `mpo_terms` compared against the sum's terms.

---

## 4. Rules a basis must follow

* **Element 1 is the identity of its space.**
  Everything that today checks for the pass-through (`ispassthrough`, `isone`, idle sites, the Jordan corners, the decay loop `λ · 𝟙`) checks for element 1 instead.
* **Charge purity.**
  Every element has one charge, which the caterpillar needs; fermion parity is part of the charge.
* **Linear independence per charge.**
  Completeness is not required: `project` throws when an operator is not in the span (with the residual), as do on-site products and `H'` when their result leaves it, with a hint to extend the basis.
* **Non-orthogonality is allowed.**
  The exact path never needs inner products; the SVD backends need the Gram matrix per charge for their truncation weights.
* **The user keeps operators and bases consistent.**
  Nothing records which basis a placed operator came from, so a mismatch gives a wrong operator, not an error.

---

## 5. Impact on the pipeline

| stage | change |
|---|---|
| letters (`IrrepOperator`) | `(charge, index)` as today, with the index into the site's basis instead of into matrix units; the pass-through sentinel becomes element 1 |
| `SiteOperator` | a plain coefficient vector over one basis |
| building terms | `*` added; `couple` re-couples out-of-order non-abelian operands with F- and R-moves (cached per shape) instead of throwing; `dot` unchanged; a `Term` gains an unevaluated on-site product node |
| inserting into the sum | product nodes resolved in the site's basis, through memoised structure constants |
| the sum | `FiniteChain`/`InfiniteChain` hold bases instead of spaces; validation only checks that each letter exists in its site's basis |
| `canonicalize!`, the term table, the sweep | **unchanged**: they compare labels |
| SVD backends | Gram matrix per charge for truncation weights; lossless results unaffected |
| output | a `SymbolicMPO`; tensors through `instantiate(entry, basis)`, MPSKit structs through `instantiate(W)` in a package extension |
| infinite chains, `expterm` | a periodic list of bases; the string operator is a symbolic operator like any other |

---

## 6. Open questions

1. **The rest of MPSKitModels.**
   Replacing it also means a home for its lattices and named models (`hubbard_model`, `heisenberg_XXZ`, …); in OpSum, or in a third package?

Decided:

* `mpo(H)` replaces `irrep_mpo`, and returns a `SymbolicMPO`; `instantiate(W)` returns the MPSKit struct; the type is called `Basis`.
* Infinite chains are spelled `InfiniteChain(bases)`.
* No basis tag on letters; mismatches are silent for now.
* `project` without a basis goes away for now.
* On-site products are written `a[i] * b[i]` and resolved on insertion; the unplaced form `(a * b)[i]` is dropped for now.
* `*` is defined whenever the overall channel is forced, which holds for every product of the neutral user-facing operators; otherwise it throws.
* `couple` and `*` accept any site order under any symmetry, re-coupling non-abelian factors with F- and R-moves.
* No separate no-allocation entry for building `N⁴` terms; symbolic operators are integer-indexed already.
* The builders (`spin`, `spin_ops`, `fermion_ops`, `matrixunit`) take a basis and return symbolic operators in it.
* Stock bases are memoised functions of the symmetry (`spin_basis`, `fermion_basis`, `hubbard_basis`, `tj_basis`, `boson_basis`), on TensorKitTensors' spaces; this package and TensorKitTensors together replace MPSKitModels.
* The charged single-site elements are obtained in OpSum by factorising TensorKitTensors' neutral two-site operators; they only define the basis, and every user-facing operator is neutral.
* Clock (ℤₙ) bases are dropped for now.

---

## 7. Staging

1. `Basis` and `project(O, basis)` on one space, with the identity at element 1; the existing sweep runs on basis indices unchanged.
2. `SymbolicMPO`, `instantiate(entry, basis)` and the MPSKit extension; the #38 numbers of §1 as the acceptance test.
3. Per-site basis lists and `vcat` for mixed lattices.
4. `*` across sites under an abelian symmetry, then on-site products with their resolution on insertion.
5. Non-abelian re-coupling in `couple` and `*`, with the spin-free chemistry Hamiltonian of §3.6 against the #38 numbers.
6. Infinite chains, `expterm` and the SVD backends.
7. The stock bases and builders, starting with spins and the Hubbard site used by #38.

---

## Appendix: related compression findings (N₂/cc-pVDZ, user basis of §1)

These are independent of the interface, but affect the same MPOs.
The experiment code is on the OpSum branch `exp/cover-tiebreak`.

**Tie-break between minimum covers.**
`min_vertex_cover_bipartite` runs König's construction from the unmatched left vertices, so wherever a choice exists it covers the prefix side.
That split the switch of the two-index channels over sites 15 and 16, leaving 4,509 stored numbers at site 16 against 634 in the hand-built MPO.
The mirror-image construction (from the unmatched right vertices) is equally minimum and puts the whole switch on one site.
Bond dimensions are identical, and stored numbers end up within 2% of the hand-built MPO.
With it as the global default the test suite passes except for 6 infinite-chain tests: 2 only reorder the bond, and 4 miss a term at the left edge of the test window (soundness holds).
So it can ship as a finite-chain policy, keeping the current tie-break on infinite chains until the window check is understood.

**Block granularity.**
At the switch site the hand-built MPO's 2× fewer Jordan entries are a counting artefact: it puts the S = 0 and S = 1 channels of an orbital pair on one bond index, and both MPOs store the same ~88k numbers.
Grouping bond indices into larger levels removes nothing without adding fill, because every channel has a distinct support pattern.
Grouping by charge, point-group irrep and *interval* (the sites that last and next act on the channel) gives 38,732 → 7,130 blocks for 1.37× the stored numbers, and 28,111 → 2,381 blocks at 1.12× at the switch site.
Making the D2h point group a real symmetry of the site spaces would give the same structure without any chemistry-specific grouping.
Whether either pays off in DMRG still needs timing with MPSKit.
