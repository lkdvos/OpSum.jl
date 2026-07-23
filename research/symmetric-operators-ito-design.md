# Design: Symmetric-tensor operators via local ITOs + fusion-tree coupling

## Context

OpSum.jl converts sums of quantum operators into MPOs/TTNOs. Today the entire numerical
substrate is dense: the local alphabet is a flat `OperatorBasis` (only `PauliOperator`
exists), `instantiate` returns dense `Array{T}`, composition is plain `kron`, and the MPO
bond optimization treats each alphabet symbol as an independent orthonormal direction. There
is no notion of symmetry, charge, or sector anywhere (no TensorKit dependency).

The goal is to make the operator system **symmetry-aware** so it can build symmetric MPOs
(and later TTNOs) directly as TensorKit `TensorMap`s. The organizing idea:

- **A local operator basis is an alphabet of irreducible tensor operators (ITOs).** Each ITO
  is fixed by a TensorKit `FusionTree` with **two uncoupled legs** (the physical out/in legs)
  fusing to **one coupled leg** — the operator charge `c` — plus **an integer** selecting
  which operator of that charge is meant.
- **How local tensors couple to neighbors is fixed by a `FusionTree`** over the sequence of
  local charges: the running fusion `c₁ ⊗ c₂ ⊗ … ⊗ c_N` whose internal edges are the virtual
  bond charges. The 1D-chain geometry fixes the tree shape (left-nested "caterpillar").

Target scope: **full non-abelian symmetries (SU(2) and beyond)** from the start. Dependency:
**TensorKit.jl throughout** the package. Materialization: **`TensorMap`-only** — the dense
`Array` path is the trivial-sector special case, retired as a separate track. Deliverable:
this architecture document (no concrete file-by-file implementation plan yet).

This document is the design; it names the current code it builds on, defines the new
abstractions, and enumerates the issues that must be resolved before implementation.

---

## 1. The two fusion-tree layers

The whole design rests on separating two independent fusion structures. Keeping them distinct
is what makes the automaton and the compression stay reusable.

### Layer (a) — the local ITO (the alphabet)

An operator on one site lives in `End(V) ≅ V ⊗ V*`, where `V` is the site's physical
`GradedSpace`. `End(V)` decomposes into irreps: `V ⊗ V* = ⊕_c (N_c copies of c)`. A single
**local ITO of charge `c`** is one component of that decomposition, represented as a TensorKit
tensor whose structural part is a `FusionTree` with 2 uncoupled legs `(phys_out, phys_in*)`
fusing to 1 coupled leg `c` (equivalently, bend one leg: a 3-leg object `c → V ⊗ V*`).

The **integer `n`** enumerates an orthonormal basis of the charge-`c` sector of `End(V)`,
`n ∈ 1:dim(Hom(V,V)_c)`. Concretely a local ITO is the pair `(c::Sector, n::Int)` **relative
to a fixed physical space `V`**.

- Spin-½ site: `V = ½`, `V⊗V* = 0 ⊕ 1` → alphabet `{𝟙 (c=0,n=1), S (c=1,n=1)}`. The rank-1
  ITO `S` carries the three components `q ∈ {−1,0,1}` as its coupled-leg index; they are *not*
  separate alphabet letters — the whole irrep is one letter.
- This generalizes `PauliOperator`'s flat `{I,X,Y,Z}`: `PauliOperator` is the trivial-sector
  (no symmetry) case where every basis element sits in charge `c = one(sector)`.

Subtlety to design around: `n` collapses two different internal multiplicities into one running
index — (i) the **fusion-vertex multiplicity** of `V⊗V*→c` (present only for non-multiplicity-
free groups) and (ii) the **degeneracy multiplicity** when `V` contains repeated irreps. For
multiplicity-free `V` and group (e.g. one spin-½ under SU(2)), `n` is trivially 1.

### Layer (b) — the bond coupling (how neighbors connect)

Along the MPO, each site injects its charge `cᵢ` into the virtual bond. The bond sector at
bond `i` is the running fusion of `c₁…cᵢ`; the term's coupling is a left-nested `FusionTree`
over `(c₁,…,c_N)` whose internal edges are the bond charges `bᵢ` and whose vertices carry
labels `μᵢ`. The chain geometry fixes the tree shape.

**This is the crux of "full non-abelian".** For abelian symmetry `bᵢ` is uniquely determined by
the prefix (charges just add) — the coupling carries no free information. For non-abelian
symmetry `a⊗b` has multiple channels, so **the same operator string coupled through different
intermediate charges is a genuinely distinct MPO degree of freedom**. That extra label must be
represented explicitly.

---

## 2. How this maps onto the current pipeline

Exploration confirmed the pipeline is cleanly layered and the symbolic/automaton/optimization
layers are largely symmetry-agnostic. The dense assumption is concentrated at two boundaries.

### Reused (nearly unchanged — operate on abstract symbols + scalar reduced coeffs)
- `Trie{K,V}` automaton — `src/datastructures/trie.jl`
- bipartite min-vertex-cover bond optimization — `src/statemachines/graphbuilding.jl`,
  `src/datastructures/bipartite.jl`
- SVD bond compression — `src/statemachines/graphbuilding.jl` (works on scalar `Matrix{T}`
  coefficient buckets)
- `VectorInterface` arithmetic over the symbolic tree — `src/operators/abstractoperators.jl`

Crucially, all of these become **block-diagonal in the bond charge**: each runs independently
per sector. That is both the correctness mechanism and the source of compression savings.

### Extension point
`OperatorBasis` (`src/operators/operatorbasis.jl`) — the ITO alphabet slots in here — plus the
single lowering hook `instantiate(b, ::Type{T}, axes)`.

### Where the dense/no-symmetry assumptions live (these change)
- `instantiate(::OperatorBasis, T, axes)` returns dense `Array` — `operatorbasis.jl:36`,
  `paulioperators.jl:25`.
- `kron`-based composition in `LocalOp`/`GlobalOp` `instantiate` and in `mpo_to_dense`
  (hardcodes `T = ComplexF64`) — `src/operators/operatoralgebra.jl`, `globalalgebra.jl`,
  `src/statemachines/state_machines.jl`.
- SVD compression buckets **one scalar matrix per distinct `Op`**, assuming each alphabet
  symbol is an independent orthonormal direction — `graphbuilding.jl:257`.
- `inner(x,y)=x==y` orthonormality — `paulioperators.jl:22`.

---

## 3. Interface redesign

### 3.1 The alphabet depends on the *space*, not just the type
This is the single biggest interface change. The current `OperatorBasis` interface is
type-level: `instances(O)`, `one(O)`, `scalartype(O)`, `namemap(O)`. But "the spin operator"
means different things on spin-½ vs spin-1 — the ITO alphabet is fixed by
`(sector type, physical GradedSpace)`. Therefore:

- `instances`, `one`, `namemap` must take the **physical space** as an argument
  (`instances(basis, V)`), not just the type.
- The `sites`/axes descriptor (currently plain `Int` dimensions passed to `instantiate`) must
  become a vector of TensorKit `GradedSpace`s (one physical space per site).

### 3.2 A new ITO `OperatorBasis`
An ITO basis element carries `(c::Sector, n::Int)` interpreted relative to a physical space.
It must implement, in space-aware form: enumeration (`instances(basis, V)`), a trivial-charge
accessor, ordering + hashing (`isless`, `hash` — used as `Trie`/`Dictionary` keys),
`scalartype`/`isreal`, and a symmetry-consistent `inner` (see §5). `namemap`-style display
generalizes to naming by `(c, n)`.

**Caveat (found in Phase 1):** the identity is *not* an alphabet letter. `id(V)` is a sum of
trivial-charge ITOs whenever `V` has degeneracy or multiple irreps; it collapses to a single
(normalized `id(V)/√dim(j')`) letter only for a multiplicity-free single-irrep space. So the
scalar identity must be handled structurally (`id(V)`), not routed through a type-level
`one(::Type{A})` basis element — see the implementation plan's Phase 2 note for the
`operatoralgebra.jl` call sites this affects.

### 3.3 `instantiate` → `TensorMap`
`instantiate(ito, sites::Vector{<:GradedSpace})` returns a TensorKit `TensorMap` (the local
tensor). `LocalOp`/`GlobalOp` composition replaces `kron` with fusion-tree-aware tensor
composition. Validation compares dense blocks of `TensorMap`s (see §7). The trivial-sector
graded space reproduces the old dense case, so `PauliOperator` survives as the no-symmetry
instance rather than a separate code path.

---

## 4. The automaton becomes charge-augmented

Because non-abelian coupling carries free information, the finite-state machine's states must
be augmented with the running bond charge (and vertex label where the group is non-mult-free).
Concretely the trie's alphabet symbol at each level becomes `(ITO, bond_charge_out, vertex)`
rather than just `ITO`. Consequences:

- Prefixes that agree on operators **and** intermediate charges share automaton states;
  differing couplings become distinct paths. This turns the single FSM into a **direct sum of
  per-sector FSMs** — the automaton is block-diagonal in the bond charge.
- The bipartite matching and the SVD compression then run **independently per bond sector**,
  which is exactly where the symmetric bond-dimension savings come from.
- Operator strings fed to the trie must be **fusion-resolved**: produced already carrying their
  caterpillar fusion tree. Enumerating the valid couplings of a user term (e.g. `S_i · S_j`)
  into total-charge-fixed strings is a new construction step ahead of `build_trie!`.

Design note: represent per-term data as `(opstring, fusiontree, reduced_coeff)` and let the
trie key encode enough of the fusion tree (bond charges + vertices) that prefixes merge
correctly. Keep the single-`V`-per-leaf `Trie` contract intact by pushing the coupling into the
key, not the value.

---

## 5. Reduced coefficients, Wigner–Eckart, and compression

By Wigner–Eckart, a symmetric MPO tensor factorizes into a **structural part** (fully fixed by
the charges/fusion trees) times a **reduced tensor** of degeneracy-space multiplicities. This is
exactly how TensorKit stores a `TensorMap` (structural fusion trees × data blocks). So:

- Building the symmetric MPO = building the **reduced** bond tensors + attaching the fusion-tree
  structure. The scalar coefficients `T` flowing through `build_trie!`/bipartite/SVD become
  **reduced matrix elements**.
- Compression stays valid **within a charge sector**, but the truncation norm must carry
  **quantum-dimension (`qdim(c)`) weighting** so that truncating reduced coefficients matches
  truncating the true operator 2-norm. TensorKit's tensor norm already includes these factors —
  route the compression norm through TensorKit-consistent inner products rather than the current
  `inner(x,y)=x==y`. This weighting matters only for non-abelian sectors but must be designed in
  now given the SU(2)+ target.
- SVD's `real(T)`/`eps(real(T))` assumptions (`graphbuilding.jl:232`) survive because reduced
  coefficients are still `<:Number`.

---

## 6. Symbolic algebra changes

- **Product of ITOs on the same site** = fusion-category recoupling (F-symbols). The `Prod`/`Pow`
  variants of `LocalOp` become fusion-aware. Can be deferred: support sums of single-ITO-per-site
  terms first and obtain on-site products through TensorKit at instantiation time.
- **Product across different sites** currently assumes commuting disjoint sites and errors on
  same-site overlap (`globalalgebra.jl` `*`, "assume commutative if sites disjoint for now").
  This becomes coupling via the bond fusion tree.
- **Coupling syntax.** Users need a way to express "couple these ITOs to a definite total
  charge" — e.g. the scalar (charge-0) contraction `S_i · S_j` of two rank-1 operators. This
  generalizes the current `op[site...]` syntax to irrep-coupled products and is a new API
  surface to design.

---

## 7. Fermions come (almost) for free

The stubbed `CommutationType`/`Commuting`/`Anticommuting` machinery (`operatorbasis.jl:22-27`,
defined but unused today) is subsumed by the sector's braiding/twist: a `FermionParity`-graded
space handles anticommutation signs automatically via TensorKit's `BraidingStyle`. This unifies
the roadmap's fermion support with the symmetry work rather than treating them separately.

---

## 8. TensorKit integration specifics (must be pinned down)

- **Duality / leg order.** An operator `V ← V` is a `TensorMap` with codomain `V`, domain `V`.
  Exposing the operator charge as one leg requires bending a leg to `c → V ⊗ V*`; get `isdual`
  and codomain/domain conventions right and consistent with the alphabet enumeration.
- **`FusionStyle`/`BraidingStyle` traits** to branch abelian (`UniqueFusion`) vs non-abelian
  (`GenericFusion`/`SimpleFusion`) handling and bosonic vs fermionic braiding.
- **No site-reordering braids** for the 1D chain: MPO left-to-right order matches site order, so
  bosonic terms need no braids; fermions rely on the graded braiding.
- **Alphabet construction**: enumerate `Hom(V,V)_c` via TensorKit `fusiontrees` / by building
  basis `TensorMap`s and reading their blocks. Decide the canonical `n`-ordering here.

---

## 9. Open questions / risks

1. **Coupling-enumeration blowup.** Enumerating all valid intermediate channels for
   long/all-to-all terms could be expensive for non-abelian groups; needs a bounded, canonical
   construction (this is the hard core of the SU(2)+ target).
2. **Trie key design.** Encoding `(ITO, bond_charge, vertex)` in the key while preserving prefix
   sharing and the `Trie{K,V}` single-value-per-leaf contract needs care (assert coupling
   invariants rather than silently deduping).
3. **Reduced-coefficient reality.** SU(2) reduced matrix elements can be kept real with the right
   conventions, but CG/F signs and TensorKit conventions may force complex `scalartype`; decide
   the convention early since it flows through the whole pipeline.
4. **`qdim` weighting correctness** in the SVD truncation — the place a subtle symmetry bug would
   hide. Validate against dense blocks explicitly.
5. **User-facing Hamiltonian API** for irrep coupling (`·` and general couplings) — ergonomics vs.
   generality.
6. **Latent cleanups to fold in**: `Sum` inner-constructor type-parameter swap
   (`abstractoperators.jl:25-26`), `mpo_to_dense` hardcoded `ComplexF64`
   (`state_machines.jl:3`). Note CLAUDE.md describes an aspirational TTNO/tree layer that does
   not yet exist in `src/` — this design is MPO-first; the bond fusion tree generalizes to a
   per-vertex fusion tree for TTNOs later.

---

## Validation strategy (for the eventual prototype)

The design is validated by a prototype, not by this document. When implemented:

- **Oracle = dense blocks.** For each symmetric `TensorMap` produced, `convert` to a dense array
  (TensorKit `convert(Array, tm)` / block reconstruction) and compare `≈` against the
  independently-instantiated dense operator — the same pattern as `test/test_mpo_bipartite.jl`
  (`mpo_to_dense(Ws, sites) ≈ instantiate(H, sites)`), now with symmetric spaces.
- **Trivial-sector regression.** A trivial (no-symmetry) `GradedSpace` must reproduce the current
  Pauli results bit-for-bit — keeps `PauliOperator` and existing tests as a regression net.
- **Symmetry checks.** Assert the built MPO tensors are actually symmetric (charge-conserving)
  and that bond dimensions are correctly resolved per sector; check SU(2) Heisenberg against a
  known-good dense/reference operator.
- **Norm consistency.** Verify the `qdim`-weighted compression norm matches the operator
  Frobenius norm on small non-abelian examples before trusting truncation.
- Run the suite with `julia --project -e 'using Pkg; Pkg.test()'`; format new code with Runic.
