# Handoff: an operator progression for the example pages

Purpose of this document: give the follow-up session a **logical progression through every operator
a user would plausibly want to build**, ordered so that each step introduces exactly one new
interface concept. The point is not coverage for its own sake — it is to read the interface's *flow*
end to end and judge where it reads well and where it grates. §6 is the watch-list for that
judgement.

Context: PR #32 (`latticeless-operators`) made operators latticeless — `Terms` is the compressible
type and the lattice is an argument to `irrep_mpo`. `research/interface-review.md` §12 records what
changed and why. The example series was *adapted* to the new API in that PR but not *extended*;
this is the extension.

## 1. What the current series covers, and what it does not

Current pages, in `docs/make.jl` reading order: `common`, `spin_chains`, `long_range`,
`ladders_and_cylinders`, `multibody`, `fermions`.

Not covered anywhere in `examples/`:

* **infinite chains** (`InfiniteChain`, generating sets) — feature #28, only in `operators.md`
* **exponentially decaying interactions** (`expterm`) — feature #29, only in `operators.md`
* **the MPSKit handoff** (`jordan_mpo_tensors` → `FiniteMPOHamiltonian` → DMRG). This is a user's
  *first* question and it is one paragraph of prose. Highest-value gap.
* **non-uniform lattices** — every example uses one space for every site
* **symmetries other than SU(2) / U(1) / Trivial / `FermionNumber`** — in particular parity-only
  fermions and non-abelian fermions
* **operators that are not Hamiltonians** — observables, correlators, charged operators
* **truncation** as a workflow (it is a subsection of `long_range`)

## 2. Verified capability probe

Before planning I probed the classes that no test or example currently exercises, so the progression
below does not propose anything out of scope. All of these ran against the PR #32 branch and were
`islossless`:

| case | result |
|---|---|
| non-uniform `FiniteChain([Va, Vb, Va, Vb])`, alternating spin-½ / spin-1 | works, `D = [4, 5, 4, 1]` |
| `Vect[FermionParity]` only (Kitaev-style pairing, U(1) broken) | works, `D = [3, 4, 3, 1]` |
| charged single-site operator (`F.cd[2]`, total ≠ unit) | `irrep_mpo` works, `D = [1,1,1,1]`; `jordan_mpo_tensors` **throws**, as documented |
| observable `Σᵢ Szᵢ` as an MPO | works, `D = [2,2,2,2,1]` |
| **`U1Irrep ⊠ SU2Irrep ⊠ FermionParity`** Hubbard site (non-abelian *and* fermionic) | works: 10-letter alphabet, `project` of a 2-site block gives 60 terms, `D = [17,18,17,1]`, lossless, tensors assemble; also works on an `InfiniteChain` (`L=1`, `D=[18]`) |

Two facts that shape the progression:

* `TensorKit.FermionSpin` is `SU2Irrep ⊠ FermionParity` — spin and parity, **no** charge. The full
  Hubbard sector has to be spelled
  `ProductSector{Tuple{U1Irrep, SU2Irrep, FermionParity}}` with labels `(n, j, p)`:
  `(0,0,0) => 1`, `(1,1//2,1) => 1`, `(2,0,0) => 1`.
* In that sector the singly-occupied sector has quantum dimension 2, so **`matrixunit` does not
  apply** and `project` is the only route. That is a genuinely good teaching moment and a place to
  check the error message is discoverable.

## 3. The progression

Each step names the *one* new interface concept, the model chosen because it is the simplest thing
that forces that concept, and the calls that appear. Code is a sketch unless marked ✅ (probed).

### Tier 0 — the shape of the whole thing

**0.1 One bond.** `dot(S[1], S[2])` on `FiniteChain(V, 2)`.
New: `Terms`, `FiniteChain`, `irrep_mpo`, `Ws, secs = mpo` destructuring, the three verification
tiers (`islossless` → `mpo_tensormap ≈ instantiate` → spectrum).
This is the "read the whole pipeline once" step; everything after adds one thing.

### Tier 1 — the on-site alphabet, abelian

**1.1 Transverse-field Ising, trivial sector.** `H = -J Σ σᶻσᶻ - h Σ σˣ` on `ℂ^2`.
New: `project` for a single site (no symmetry to exploit, so the Pauli matrices are just written
down), and **mixing K=1 and K=2 terms in one operator**.

**1.2 XXZ, U(1).** `matrixunit`, then `spin_ops`.
New: abelian charges; a **composite** on-site operator (`Sᶻ` is two letters); `couple` distributing
over both operands without the user expanding anything.

### Tier 2 — non-abelian

**2.1 Heisenberg, SU(2).** `spin`, `dot`.
New: `dot` versus `couple`, and why `dot` carries the `-√dim(c)` Cartesian factor and therefore does
not distribute.

**2.2 Biquadratic / AKLT, spin-1.** `H = Σ [S·S + β (S·S)²]`.
New: **there is no symbolic on-site product**, so `(S·S)²` must be built as a `TensorMap` and
`project`ed. This is the best motivation in the whole series for `project` of a K-site block, and it
arrives from physics rather than from API exposition.

### Tier 3 — arity and fusion channels

**3.1 Three-body chirality**, `(S₁ × S₂)·S₃`: nested `couple` with `to`.
New: intermediate fusion channels as explicit physical labels; why the variadic form throws
non-abelian.

**3.2 Four-body / ring exchange**: two inner lines, channel enumeration, mutual orthogonality.
(Largely the existing `multibody` page.)

**3.3 Abelian variadic**: `couple(cd[1], c[2], cd[3], c[4])`.
New: when every intermediate charge is forced there is nothing to name, so the whole chain folds.
Put this next to 3.1 so the asymmetry is visible in one screen — see watch-list item W5.

### Tier 4 — fermions

**4.1 Free hopping**, `FermionNumber`: `fermion_ops`, out-of-order `couple`, `adjoint(h, lat)`.
New: fermionic braiding, who supplies the sign, **no Jordan–Wigner strings**.

**4.2 Kitaev chain**, `Vect[FermionParity]` ✅: `−t(c†ᵢcᵢ₊₁ + h.c.) + Δ(c†ᵢc†ᵢ₊₁ + h.c.)`.
New: a symmetry that is **not** U(1)-graded — pairing breaks charge but preserves parity. Shows the
symmetry is the user's choice, not the package's.

**4.3 Hubbard, spin-orbital encoding**: one orbital per site, on-site `U n↑n↓` as a two-site term.
(Existing page.)

**4.4 Hubbard with SU(2) spin**, `U1Irrep ⊠ SU2Irrep ⊠ FermionParity` ✅.
New: **non-abelian and fermionic at once** — the hardest case, and verified in scope. `matrixunit`
is unavailable (spin-½ sector has dim 2), so the route is `project` of the bond block. Closes the
loop with 2.2.

### Tier 5 — the lattice itself

**5.1 Alternating spin-½ / spin-1 chain** ✅: `FiniteChain([Va, Vb, Va, Vb])`.
New: a **non-uniform** lattice, and therefore that `FiniteChain(V, N)` is only the convenience
constructor. Nothing in the series shows this today. See watch-list W7.

**5.2 An impurity site**: one different space in an otherwise uniform chain — the practical version
of 5.1.

### Tier 6 — geometry

**6.1 Ladder and cylinder**: column-major ordering, `cylinder_bonds`, D tracking the circumference
and not the length. (Existing page.)

### Tier 7 — long range and truncation

**7.1 Haldane–Shastry / power law**: all-to-all, D linear in N, `opsum`'s one-pass cost.
**7.2 Truncation as a workflow**: `SVDBondAlgorithm`, `IndependentSVD` vs `SequentialSVD`, and the
fact that `islossless` says nothing once truncation bites — the honest check is operator error.
Promote out of `long_range` into its own section or page.

### Tier 8 — infinite chains

**8.1 Infinite Heisenberg, `L = 1`**: `InfiniteChain`, the **generating set** semantics, and the
translation-class double-count rejection (writing both `dot(S[1],S[2])` and `dot(S[2],S[3])` on a
one-site cell).
**8.2 Dimerised chain, `L = 2`**: why a bigger cell, and alternating couplings.
**8.3 Exponential decay**: `expterm`, `decay` counted per site, `exitsite`, `string`, a sum of
exponentials, and the period difference between `FiniteChain` and `InfiniteChain`.

### Tier 9 — operators that are not Hamiltonians

This tier does not exist today and is where the interface is least exercised.

**9.1 Total magnetisation / particle number** ✅: `opsum(S.Sz[i] for i in 1:N)` — a pure K=1 sum
compressed as an MPO for measurement.
**9.2 A two-point correlator**: a single `dot(S[i], S[j])` term as a one-term MPO.
**9.3 A charged operator** ✅: `F.cd[2]`, total charge ≠ unit.
New: charged totals are first-class for `irrep_mpo`, **but `jordan_mpo_tensors` refuses them**
(a Jordan MPO's right boundary is an identity). So the measurement path and the Hamiltonian path
diverge here — worth stating plainly rather than letting a user discover it. See W6.
**9.4 A string order parameter**: a single term with a run of active sites.

### Tier 10 — handing the MPO onward

**10.1 `irrep_mpo_tensors`** → contract with `mpo_tensormap` → compare against `instantiate`.
**10.2 `jordan_mpo_tensors` → MPSKit**: `FiniteMPOHamiltonian`, then a DMRG ground state checked
against exact diagonalisation. The single most valuable missing page.
**10.3 The infinite handoff**: `irrep_mpo_tensors(Hinf, chain)` and the tiling property
`space(Ts[1], 1) == space(Ts[L], 4)'`. Confirm what MPSKit accepts here before promising anything.

## 4. Suggested page layout

Nine pages instead of six. Reading order for `docs/make.jl`:

| file | tiers | status |
|---|---|---|
| `common.jl` | scaffolding | exists |
| `getting_started.jl` | 0, 1 | **new** |
| `spin_chains.jl` | 2 | exists, add 2.2 |
| `multibody.jl` | 3 | exists, add 3.3 |
| `fermions.jl` | 4 | exists, add 4.2, 4.4 |
| `lattices.jl` | 5, 6 | **new** (absorbs `ladders_and_cylinders`) |
| `long_range.jl` | 7 | exists, promote 7.2 |
| `infinite.jl` | 8 | **new** |
| `observables.jl` | 9 | **new** |
| `mpskit.jl` | 10 | **new** |

Splitting 8 into `infinite.jl` + `exponential_decay.jl` is also defensible — `expterm` has enough
surface (decay, exitsite, string, sums, period) to carry a page.

## 5. Mechanics

* Pages are Literate sources in `examples/`, rendered by `docs/make.jl`; each new file must be added
  to the `EXAMPLES` vector there, **in reading order** (it sets the nav order).
* Each page opens with
  `using OpSum: OpSum` then `include(joinpath(pkgdir(OpSum), "examples", "common.jl"))`.
* `common.jl` provides `build(name, h, lat; alg, quiet)`, `mpo_matches_oracle(h, lat)`,
  `spectrum(h, lat)`, `hermiticity_error(h, lat)`, `cylinder_bonds`, `ladder_bonds`, and the
  bond-dimension reporters. All take `(h, lat)` as separate arguments since PR #32.
* `test/test_showcase_models.jl` pins the bond dimensions of the benchmark models; if a page
  introduces a model worth pinning, add it to `benchmark/ShowcaseModels.jl` (builders return
  `(h, lat)`) rather than duplicating it.
* Verify with `julia --project=docs docs/make.jl` — it runs Literate over every page **and**
  executes every `@example` block in `operators.md`, so it is the real gate. Budget ~10 min.
* Keep pages cheap: `instantiate` is exponential in N, and contracting an N-site MPO costs a fresh
  `ncon` specialisation per `(length, sectortype)`. The existing pages spend the dense oracle only
  where a sign or a fusion coefficient could be wrong.

## 6. Watch-list: what to judge while reading the flow

The point of the exercise. Each of these is a place I expect friction; the progression should make
it possible to decide rather than guess.

* **W1 — lattice repetition.** Option A's cost. Count how often `lat` is repeated on one page:
  `build(name, h, lat)`, `islossless(h, lat)`, `instantiate(h, lat)`, `irrep_mpo(h, lat)`. The
  per-page `chain(N) = FiniteChain(V, N)` helper is the current mitigation. If it still reads as
  noise, the fix is a caller-side pair (as `benchmark/ShowcaseModels.jl` now does), not a return to
  lattice-bound operators.
* **W2 — losing `h'`.** `adjoint(h, lat)` appears in Tier 4. Judge how much worse
  `T + adjoint(T, lat)` reads than `T + T'`, now that it is in front of you in a real model.
* **W3 — five routes to an on-site operator**: `matrixunit`, `spin`, `spin_ops`, `fermion_ops`,
  `project`. At each step, is the right choice obvious *without* reading the reference? Tier 4.4 is
  the sharp case: `matrixunit` silently does not apply when a sector has dim > 1 — check the error
  is discoverable.
* **W4 — builder naming.** `spin(V)` returns one operator; `spin_ops(V, secs)` and `fermion_ops(V)`
  return NamedTuples. Reading 2.1 and 1.2 back to back should settle whether to unify on
  `spin_ops(V)` for every symmetry.
* **W5 — abelian variadic vs non-abelian nesting.** 3.1 and 3.3 adjacent. Decide whether the
  single-spelling `couple(ops...; channels = …)` idea from `interface-review.md` §7 is worth it, and
  whether a `couple_channels` query should exist (the current `multibody` page discovers the legal
  channels with a `try`/`catch` loop, which is evidence something is missing).
* **W6 — Hamiltonian path vs measurement path.** Tier 9 is the first place charged operators and the
  `jordan_mpo_tensors` restriction meet. Does the interface make that divergence clear, or does a
  user only find out by hitting the throw?
* **W7 — non-uniform lattice ergonomics.** 5.1 currently reads
  `FiniteChain(repeat([Va, Vb], ncells))`. `InfiniteChain([Va, Vb])` names a *cell*; a
  `FiniteChain(cell, ncells)` constructor would mirror it. Worth deciding once a real alternating
  model is on the page.
* **W8 — `project`'s full-support wart.** Every projected term is active on all `K` sites, so
  `S₁·S₂ + ¼` comes back as two two-site terms. 2.2 and 4.4 both go through `project`; judge whether
  the note in `operators.md` is enough.
* **W9 — `dot` needs `using LinearAlgebra: dot`.** Appears on nearly every page. Either accept and
  document once, or reconsider.

## 7. Open questions for the author

1. Is the nine-page layout right, or should the series stay shorter and denser?
2. Tier 9 (observables) is new territory — is measuring operators in scope for OpSum's docs, or is
   the package deliberately Hamiltonian-only? This decides whether §9 is a page or a paragraph.
3. Should `benchmark/ShowcaseModels.jl` grow to cover the new models (Kitaev, SU(2) Hubbard,
   non-uniform), so their bond dimensions are pinned by `test_showcase_models.jl`?
4. For 10.3, what does MPSKit currently accept for infinite MPO Hamiltonians? Confirm before
   writing the page.
