# Phased implementation plan: symmetric operators via ITOs

Companion to [`symmetric-operators-ito-design.md`](./symmetric-operators-ito-design.md). That
document is the architecture; this one is the build order. The guiding principle is
**incremental, always-green**: the existing dense/Pauli pipeline stays as a regression net, and
each phase is independently testable. We deliberately **start by prototyping the alphabet layer**
(Phase 1) in isolation, before touching the automaton or the MPO assembly.

Target: full non-abelian (SU(2)+), TensorKit 0.17 throughout, `TensorMap`-only materialization.

## Environment (verified)
- Installed: **TensorKit 0.17.1**, **TensorKitSectors 0.3.9** (re-exported by TensorKit).
- API facts this plan relies on (confirmed against the installed sources):
  - Trivial sector is **`unit(I)`** (`one`/`isone` are aliases). Fusion of sectors: `a ⊗ b`
    (unique outputs), multiplicities via `Nsymbol(a,b,c)`.
  - Spaces: `Vect[I](c => deg, …)`, aliases `SU2Space`, `Rep[U₁]`, `Z2Space`, `ComplexSpace`/`ℂ^d`.
    `dim(V, c)` = **degeneracy** of `c` in `V`; `dim(c)` = **quantum dimension**; `reduceddim(V)`
    = Σ degeneracies; `dual(V)`/`V'`, `sectors(V)`, `flip`.
  - `FusionTree{I,N,M,L}` + `fusiontrees(uncoupled, coupled, isdual)` iterator.
  - `TensorMap`: `zeros(T, W ← V)`, reduced block **`t[f₁, f₂]`** (`f₁`=codomain splitting tree,
    `f₂`=domain fusion tree) returns a **view** into the degeneracy data; `block(t,c)`,
    `blocks(t)`, `blocksectors(t)`, `space/codomain/domain/numout/numin`.
  - Densify with **`convert(Array, t)`** (no `Array(t)` method). `norm(t)` and `inner(t1,t2)` are
    **qdim-weighted** (`Σ_c dim(c)·…`) — this is exactly the truncation norm the design needs.
  - Leg bending: `permute`, `repartition`, `twist` (fermionic legs), `flip`, `braid`.

---

## Phase 0 — Dependency + scaffolding (small, mechanical)

- Add `TensorKit = "0.17"` to `Project.toml` `[deps]`/`[compat]`. Confirm load + precompile.
- Create a submodule for the symmetric basis, mirroring `PauliOperators`
  (`src/operators/paulioperators.jl`): new file `src/operators/irreptensoroperators.jl`, `include`d
  from `src/OpSum.jl` after `operatorbasis.jl`. Empty stub + `using TensorKit` first.
- No behaviour change yet; existing tests must stay green.

**Done when:** package loads with TensorKit; `Pkg.test()` unchanged.

---

## Phase 1 — Alphabet layer prototype ★ (start here)

Goal: a standalone, tested **local ITO alphabet** for a fixed physical `GradedSpace` — enumerate,
materialize to `TensorMap`s, with a canonical `(c, n)` labeling that is orthonormal under
TensorKit's `inner`, and that reproduces known cases. This phase touches **only** the new basis
type + the `OperatorBasis` interface; it does not require the automaton or MPO code.

### 1a. Basis element type
```julia
struct IrrepOperator{I<:Sector} <: OperatorBasis
    c::I      # operator charge (coupled leg)
    n::Int    # which operator of that charge (canonical index)
end
```
The element is defined **relative to a physical space `V`**, which is supplied by context (as
`sites` already is), not stored in the element. This keeps trie/dictionary keys small.

**Decision point:** whether `n` is a single flat index or a structured `(blocksector, tree, deg)`
tuple. Recommend flat `Int` with a documented canonical ordering (below), tuple only internally.

### 1b. Space-aware interface (generalizes `operatorbasis.jl`)
The single biggest interface change from the design (§3.1): make enumeration/identity space-aware.
- `instances(::Type{IrrepOperator}, V)` → `Vector{IrrepOperator{I}}` of all `(c,n)` spanning
  `End(V)`.
- identity ITO: `one(::Type{IrrepOperator}, V)` = the `c = unit(I)` operator equal to `id(V)`.
- `Base.isless` / `Base.hash` on `(c,n)` (for `Trie`/`Dictionary` keys and `sortkeys!`). Requires a
  total order on sectors — **decision point:** define via TensorKit's canonical `SectorValues`
  iteration order, tie-broken by `n`.
- `scalartype`/`isreal` (SU(2) reduced elements can be real with the right convention — see design
  risk #3; pick the convention here).

### 1c. Materialization
`instantiate(op::IrrepOperator, V)` → `TensorMap` in the canonical form
```
O_{c,n} :  V  ←  V ⊗ Vect[I](c => 1)
```
i.e. an operator `V←V` with a dangling charge-`c` leg of degeneracy 1. When `c == unit(I)` the
charge leg is trivial and this is just an operator `V←V`, recovering the dense case. Build by
`zeros(T, V ← V ⊗ Cc)` and setting the `n`-th reduced entry (via `O[f₁, f₂]` views) to the
normalization constant.

- **Canonical `n` ordering:** flatten over (coupled/blocksector `j'`, domain fusion tree `f₂` of
  `(j, c) → j'`, degeneracy indices `1:dim(V,j') × 1:dim(V,j)`). Document precisely — everything
  downstream keys on it.
- **Normalization (decision point):** choose the per-entry constant so that `instances(…, V)` is
  orthonormal under TensorKit's qdim-weighted `inner`. Then `inner(x::IrrepOperator,
  y::IrrepOperator)` collapses to `x == y` again, keeping the existing `Sum`-inner machinery cheap
  (`abstractoperators.jl:82-123`).

### 1d. Display
`show`/`namemap`-style naming by `(c, n)` (e.g. `S⁽¹⁾`, `𝟙`). Generalizes the finite `namemap`.

### 1e. Tests (put in `test/`, e.g. `test_irrep_alphabet.jl`)
- **SU(2) spin-½** (`V = SU2Space(1//2 => 1)`): `instances` == `{(0,1), (1,1)}`;
  `convert(Array, instantiate((1,1), V))` reproduces the spin-1 spherical-tensor components and
  relates to `S_x,S_y,S_z` / Pauli; `(0,1)` is the (normalized) identity.
- **U(1)** toy (e.g. `Rep[U₁]` hardcore boson / spinful site): charges enumerate correctly; ladder
  operators land in the expected charge sectors.
- **Trivial sector** (`V = ℂ^2`): 4 ITOs spanning `End(ℂ^2)`; `project` of `σx/σy/σz` round-trips;
  reproduces the current Pauli results (regression bridge).
- **Orthonormality:** `inner(instantiate(xᵢ,V), instantiate(xⱼ,V)) ≈ δᵢⱼ` (validates normalization
  + qdim weighting).
- **Completeness:** projecting an arbitrary operator onto the alphabet preserves `norm`.

**Done when:** the alphabet can be enumerated, materialized, and validated for SU(2)/U(1)/trivial
independently of any MPO machinery. This is the prototype requested as the starting point.

---

## Phase 2 — Symbolic layer over ITOs

- Make `LocalOp{T,A}` carry `A = IrrepOperator`. `Sum`/scalar arithmetic works via existing
  `VectorInterface` paths. **On-site products** (`Prod`/`Pow`) = fusion recoupling — **defer**:
  support sums of single-ITO-per-site terms first (design §6); reach on-site products through
  TensorKit at instantiation later.
- `instantiate(::LocalOp, V)` composes ITO `TensorMap`s (`Sum` → add TensorMaps; scalar → `scale`).
  Replace the `kron` path (`operatoralgebra.jl:52-54`) with TensorKit composition.
- `GlobalOp`: `op[site...]` with per-site spaces; `instantiate(::GlobalOp, sites::Vector{<:GradedSpace})`
  → global `TensorMap` oracle (replaces the `mapfoldl(kron, …)` embedding in `globalalgebra.jl`).
- **Coupling API (decision point):** a way to express irrep-coupled products, e.g. `S_i · S_j`
  (charge-0 contraction of two rank-1 ITOs). Generalizes `op[site...]`; design the surface here.
- **★ The identity is NOT an alphabet letter — rework the scalar-identity handling (surfaced in
  Phase 1).** The current `operatoralgebra.jl` routes the scalar identity *through* a type-level
  basis-element identity `one(::Type{A})`, which does not exist and cannot exist for `IrrepOperator`
  (it needs the physical space `V`), and which is not a single letter anyway:
  - `id(V) = Σ_{j′} Σ_row E_{j′,row,row}` is a **sum** of trivial-charge ITOs whenever `V` has
    degeneracy or multiple irreps (`ℂ^2`, boson sites, spin-½⊕spin-1, …). It collapses to a single
    ITO only for a multiplicity-free single-irrep space (one SU(2) spin), and even then it is the
    **normalized** `id(V)/√dim(j′)`, not `id(V)`. Phase 1's `one(::Type{<:IrrepOperator}, V)` is a
    placeholder correct only in that multiplicity-free case.
  - Fix the two concrete call sites that assume `one(::Type{A})`:
    - `instantiate` of a scalar variant (`operatoralgebra.jl:40`, `instantiate(o * one(A), sites)`):
      map the scalar `c` **structurally** to `c * id(V)` via TensorKit's `id`, not via `one(A)`.
    - `operatorstrings` (`operatoralgebra.jl:225`) and idle-site Kron padding: idle sites must be
      filled with a **distinguished structural "pass-through" identity symbol** (injects trivial
      charge `unit(I)`, acts as `id(V)`), *not* one of the enumerated `(c,n)` letters. This carries
      through to the Phase 3 trie alphabet (the pass-through symbol is separate from ITO letters).
  - The generic `OperatorBasis` methods that assume a type-level identity (`one(x)=one(typeof(x))`,
    `zero(x)=0*one(x)`, `inner(x::OperatorBasis, ::Number)`) in `operatorbasis.jl` currently *throw*
    for `IrrepOperator` (no `one(::Type)`); either make them space-aware or funnel all `IrrepOperator`
    arithmetic through `LocalOp`, where the identity lives as the **scalar** `one(T)`, not `one(A)`.
  - If identity is ever expressed in the alphabet basis, the `√dim` normalization factor must be
    tracked consistently into the reduced-coefficient buckets (Phase 4).

**Done when:** SU(2) two-site terms (e.g. `S_i·S_j`) instantiate to correct dense operators via the
`TensorMap` path, matching an independent dense construction.

---

## Phase 3 — Charge-augmented automaton (fusion-resolved terms)

- Term normal form: `(opstring::Vector{IrrepOperator}, fusiontree_over_charges, reduced_coeff::T)`.
- **Fusion-resolve** user terms into normal form: enumerate coupling channels (intermediate bond
  charges) for the term's total charge. This is the hard core of the SU(2)+ target (design risk #1)
  — needs a bounded, canonical enumeration.
- Augment the `Trie` key: symbol = `(IrrepOperator, bond_charge_out[, vertex])`. Adapt
  `build_trie!`/`operatorstrings` (`globalalgebra.jl:388-504`, `operatoralgebra.jl:224-255`).
- Preserve the `Trie{K,V}` single-value-per-leaf contract — push coupling into the **key**, not the
  value; **assert** the coupling invariants rather than silently deduping.

**Done when:** the trie for an SU(2) Heisenberg chain is charge-block-diagonal and round-trips
(reconstructing the term list from trie paths).

**Status: implemented** — `src/operators/irreptrie.jl` + `test/test_irrep_trie.jl` (full suite
794 green). `FusedTerm` normal form, `ITOKey = (op, bond, vertex)` trie key, pass-through
identity sentinel (`IrrepOperator(unit(I), 0)`), channel enumeration via TensorKit `fusiontrees`.
**Caveat:** validation is *structural only* (block-diagonality + trie↔term round-trip); that a
`FusedTerm` re-materializes to the correct dense operator is deferred to Phase 5's oracle, so the
reduced-coefficient convention is not yet independently verified end-to-end. Deferred within
Phase 3: GenericFusion vertex multiplicity > 1 and multi-body (>2-operator) coupling.

---

## Phase 4 — Per-sector bond optimization + compression

- Run the bipartite min-vertex-cover matching (`graphbuilding.jl`, `bipartite.jl`) and the SVD
  compression **independently per bond charge sector** (block-diagonal). The scalar `Matrix{T}`
  coefficient buckets (`graphbuilding.jl:220-276`) become per-`(sector, operator)`.
- Truncation norm: reuse TensorKit's already-qdim-weighted `norm`/`inner`, or replicate the
  `dim(c)` factor in the reduced-coefficient SVD so truncation matches the true operator 2-norm
  (design §5, risk #4). `real(T)`/`eps(real(T))` assumptions survive.

**Done when:** per-sector bond dimensions match hand-computed values on small SU(2)/U(1) models, and
compression is a faithful (validated) truncation.

---

## Phase 5 — Symmetric MPO assembly + end-to-end validation

- Assemble per-site MPO tensors as `TensorMap`s from the reduced bond tensors + fusion structure;
  bond legs become `GradedSpace`s built from the per-sector multiplicities.
- Replace `mpo_to_dense` (`state_machines.jl`, hardcoded `ComplexF64`): densify via
  `convert(Array, tensor)` and contract; drop the hardcoded scalar type.
- End-to-end: build the **SU(2) Heisenberg** MPO; validate `≈` against a dense reference (design §7
  validation); check per-sector bond dims; confirm tensors are charge-conserving.

**Done when:** symmetric MPOs validate against dense oracles for SU(2)/U(1), and the trivial-sector
path reproduces the current `test_mpo_bipartite.jl`/`test_mpo_svd.jl` results.

---

## Cross-cutting

- **Fermions** fall out at Phases 1–2: use a `FermionParity`-graded `V`; leg bending in `instantiate`
  applies `twist` on fermionic legs (API §6). The stubbed `CommutationType` (`operatorbasis.jl:22-27`)
  becomes redundant — the sector's `BraidingStyle`/`twist` carries the signs (design §7).
- **Latent cleanups** to fold in opportunistically: `Sum` inner-constructor type-parameter swap
  (`abstractoperators.jl:25-26`); CLAUDE.md's aspirational TTNO/tree layer does not exist yet — this
  plan is MPO-first, TTNO is a later generalization (per-vertex fusion trees).
- **Ordering of value:** Phases 1–2 already give a usable "symmetric operator algebra + dense
  oracle" even before the MPO machinery; Phases 3–5 add the compressed symmetric MPO.

## Suggested first commit boundary
Phase 0 + Phase 1 together: dependency + a fully-tested standalone ITO alphabet. That is the
smallest independently-valuable, independently-verifiable unit.
