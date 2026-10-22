# Building operators

A short guide to the operator interface.
The example pages (starting with [Spin chains](@ref)) work through concrete models; [Reference](@ref) lists every docstring.

```
TensorMap ──project(O, B)──► SiteOperator / LocalOperator ──A[i], B[i, j]──► Terms ──opsum(lat, …)──► OperatorSum ──mpo──► SymbolicMPO
spin_ops, fermion_ops, … ───┘
```

A **site** is an operator [`Basis`](@ref OpSum.Basis): an ordered list of operators on the physical space, identity first.
A bag of [`Terms`](@ref OpSum.Terms) carries no lattice; the lattice (one basis per site) enters when terms are added to an [`OperatorSum`](@ref OpSum.OperatorSum), the one container that `mpo`, `instantiate` and `islossless` take.
Everything is exported except `dot` (`using LinearAlgebra: dot`).

!!! warning "Never write alphabet letters by index"
    Behind every operator are letters `Letter(c, n)`, the `n`-th basis element of charge `c`.
    Their order depends on how the basis was built, so hard-coding an `n` silently changes the Hamiltonian when the basis changes.
    Use builder entries or `project` and let OpSum find the letters.

## On-site operators

Each common site has a basis (`spin_basis`, `fermion_basis`, `boson_basis`, `hubbard_basis`, `tj_basis`; the symmetry is an argument) and a builder of the same stem (`spin_ops`, `fermion_ops`, `boson_ops`, `hubbard_ops`, `tj_ops`).
A builder returns a `NamedTuple` of every charge-neutral operator the symmetry admits, expanded in the basis and named after [TensorKitTensors.jl](https://github.com/QuantumKitHub/TensorKitTensors.jl).
Single-site entries are [`SiteOperator`](@ref OpSum.SiteOperator)s, multi-site entries unplaced [`LocalOperator`](@ref OpSum.LocalOperator)s.
The stock bases are complete, so products and `H'` never leave them.

```@example ops
using OpSum, TensorKit

B = spin_basis(U1Irrep)
ops = spin_ops(U1Irrep)
keys(ops)
```

Any other operator is a `TensorMap` handed to [`project`](@ref OpSum.project); the expansion is exact and checked, and it throws if the operator is outside the span of the basis.

```@example ops
Bc = spin_basis()                  # no symmetry
Vc = space(Bc)
σx = project(TensorMap(ComplexF64[0 1; 1 0], Vc ← Vc), Bc)
```

`Basis(ops; complete = true)` builds a basis of your own from neutral operators.
Operators that appear in your Hamiltonian should be elements of the basis: the same model in a poorer basis (e.g. matrix units) has more letters per term and a larger bond dimension.
`scalarop(c, I)` is a multiple of the identity, which is a pass-through and dropped on placement.

## Placing and multiplying

`A[i]` places a `SiteOperator`; `B[i, j]` places a `LocalOperator` on explicit increasing sites, `B[i]` on `i:i+K-1`.
Sites need not be adjacent.
`*` between placed operators is the operator product in the order written.

```@example ops
term = ops.S_plus_S_min[2, 5]
szz = ops.S_z[1] * ops.S_z[2]
szz ≈ ops.S_z_S_z[1, 2]
```

* Site order is free: out-of-order terms are braided into site order (for fermions, the anticommutation sign; `f_plus_f_min[i + 1, i]` is written as it reads).
* A product on a repeated site is resolved in the site's basis when it enters an `OperatorSum` (until then `length` and `≈` on the bag throw).
* `*` needs the factors' total charges to fuse to one sector, which neutral operators always do.

```@example ops
ex = spin_ops(SU2Irrep).S_exchange
three = ex[1, 2] * ex[2, 3]          # shared site multiplied in the basis
four = ex[1, 3] * ex[2, 4]           # interleaved sites
(typeof(three), typeof(four))
```

`LocalOperator`s add, subtract and scale, and `B + α` adds a pass-through slot.
Placed and unplaced operands cannot be mixed in one call.

### Projecting a block

A `K`-site `TensorMap` (neutral, or with a trailing `Vect[I](tot => 1)` leg) is projected into a basis, or a list of bases, one per slot, giving a `LocalOperator` to place with `[i]`.

```@example ops
Bu = spin_basis(U1Irrep)
hbond = instantiate(
    opsum(
        FiniteChain(Bu, 2),
        (1 / 2) * ops.S_plus_S_min[1, 2] + (1 / 2) * ops.S_min_S_plus[1, 2] + ops.S_z_S_z[1, 2],
    )
)
Bxxz = project(hbond, Bu)
```

### Charged operators: `couple` and `dot`

[`couple`](@ref OpSum.couple) fuses placed charged operators to a total charge (default the unit sector), and `dot` is the Cartesian `Sᵢ·Sⱼ`.
For three or more operands under a non-abelian symmetry, nest `couple(couple(a, b; to = j), c)` or list the options with `couple_channels`.

```@example ops
Bs = spin_basis(SU2Irrep)
Sv = project(Bs[2], Bs)                     # the spin vector: the charge-1 element
chirality = couple(Sv[1], Sv[2], Sv[3])
```

## OperatorSum and the hermitian conjugate

`opsum(lat, terms...)` makes one pass over each argument (iterables nest) and checks every letter and site as it is added.
`opsum!(H, terms...)` / `append!` mutate; `+` copies, so do not fold it over thousands of terms.
A lattice is `FiniteChain(B, N)`, `FiniteChain([B₁, B₂, …])`, `InfiniteChain(B)` or `InfiniteChain([B₁, B₂])`.
`H'` builds the hermitian conjugate, so a hopping only has to be written once; it needs a lattice, so a bare `Terms` has none.

```@example ops
Bf = fermion_basis(U1Irrep)
F = fermion_ops(U1Irrep)
flat = FiniteChain(Bf, 2)
T = opsum(flat, -1.0 * F.f_plus_f_min[1, 2])
T + T' ≈ opsum(flat, -1.0 * (F.f_plus_f_min[1, 2] + F.f_plus_f_min[2, 1]))
```

```@example ops
Hxxz = opsum(FiniteChain(Bu, 6), Bxxz[i] for i in 1:5)
length(Hxxz)
```

## Building the MPO

```@example ops
W = mpo(Hxxz)
map(length, W.bondsectors)       # reduced bond dimension per bond
```

`W.Ws[i]` is the sparse bond matrix at site `i` and `W.bondsectors[i]` the charge of each bond index to its right.
`mpo_tensors(W)` assembles symmetric `TensorMap`s.
The optional second argument picks the bond algorithm: `VertexCover()` (default, lossless minimum vertex cover) or `IndependentSVD(trunc)` / `SequentialSVD(trunc)` with a `MatrixAlgebraKit` truncation strategy.
`IndependentSVD` keeps `k` indices per bond; `SequentialSVD` keeps `k` after upstream truncation.

```julia
mpo(H, IndependentSVD(truncrank(8)))
mpo(H, SequentialSVD(truncrank(8)))
```

[`jordan_mpo_tensors`](@ref OpSum.jordan_mpo_tensors) emits one `BlockTensorKit.SparseBlockTensorMap` per site in Jordan form, and with MPSKit loaded `instantiate(mpo(H))` builds its `FiniteMPOHamiltonian` / `InfiniteMPOHamiltonian` (see the MPSKit example page).
Jordan emission requires total charge `unit(I)`; operators you measure (magnetisation, correlators) compress by the same `mpo` call, but a charged one is refused by `jordan_mpo_tensors`.

```@example ops
Wsj = jordan_mpo_tensors(Hxxz)
map(W -> size(W, 4), Wsj)
```

Checks, cheapest first: `islossless(H)` (symbolic, any `N`, assumes lossless compression); `mpo_tensormap(mpo_tensors(W)) ≈ instantiate(H)` (dense oracle); and a spectrum comparison of `instantiate(H)` (`examples/common.jl` has helpers).

```@example ops
islossless(Hxxz), mpo_tensormap(mpo_tensors(W)) ≈ instantiate(Hxxz)
```

## Infinite chains

On an `InfiniteChain` the same terms give an infinite MPO with a unit cell of `length(chain)` sites.
The terms are a generating set, one representative per translation class: the operator is ``\sum_n \mathrm{translate}(H, nL)``.
Terms must be neutral and non-empty (`K ≥ 1`), and the SVD strategies are not available.

```@example ops
chain = InfiniteChain(Bu)
Hgen = opsum(
    chain, 0.5 * ops.S_plus_S_min[1, 2] + 0.5 * ops.S_min_S_plus[1, 2] + ops.S_z[1] * ops.S_z[2]
)
Hinf = mpo(Hgen)
Hinf.bondsectors, Hinf.start, Hinf.finish
```

## Exponentially decaying interactions

[`expterm`](@ref OpSum.expterm) adds ``\sum_{i<j} \lambda^{j-i-1} A_i S_{i+1}\cdots S_{j-1} B_j`` with ``0 < |\lambda| < 1``, from one representative term; it goes into the same `OperatorSum` on either lattice.

```@example ops
Hexp = opsum(
    chain,
    0.5 * ops.S_plus_S_min[1, 2] + 0.5 * ops.S_min_S_plus[1, 2],
    expterm(ops.S_z_S_z[1, 2]; decay = 0.6),
)
mpo(Hexp).bondsectors
```

* `decay` counts per site of the string; `|λ| ≥ 1` is rejected.
* `exitsite` (default: the representative's last site) is the first site of the exit block.
* `string` is a neutral on-site operator carried by the gap (default: identity); the sites a channel touches must share one basis.
* A sum of exponentials is a sum of `expterm`s, one bond index per distinct `λ`.
* On a finite chain the sum is truncated to the chain (`OpSum.chain_terms(H)` spells it out). Use the smallest unit cell you can, since a channel's period is part of its declaration.

## Time evolution

[`taylor_mpo`](@ref OpSum.taylor_mpo)`(H, N)` builds an MPO of ``U \approx e^{\tau H}`` that is correct through ``\tau^N``, following Van Damme, Haegeman, McCulloch and Vanderstraeten, [SciPost Phys. 17, 135 (2024)](https://doi.org/10.21468/SciPostPhys.17.5.135) ("Efficient higher-order matrix product operators for time evolution").
`H` is an `OperatorSum` (or a [`SymbolicMPO`](@ref OpSum.SymbolicMPO)); the result is a [`TaylorMPO`](@ref OpSum.TaylorMPO), with the same bond sectors as a `SymbolicMPO` but with the time left **symbolic**: every entry is a polynomial in ``\tau``.
The convention is ``U \approx e^{\tau H}``: imaginary time ``e^{-\beta H}`` is ``\tau = -\beta``, real time ``e^{-itH}`` is ``\tau = -it``.

```@example ops
U = taylor_mpo(Hxxz, 2)
```

Evaluating at a number gives the numeric bond matrices (sparse matrices of `SiteOperator`s), and `mpo_tensors(U, τ)` the symmetric site tensors.

```@example ops
Ws = U(-0.1)                      # = evaluate(U, -0.1): bond matrices at τ = -0.1
Ts = mpo_tensors(U, -0.1)
```

The error against ``e^{\tau H}`` scales as ``\tau^{N+1}``, with the MPO size-extensive (it contains all disconnected products of terms of `H`).

* `extension = true` adds the terms of ``H^{N+1}`` that overlap only ``N`` times, for a smaller error at little or no cost in bond dimension (none for the Ising chain; `16, 52, 155` instead of `14, 42, 117` for `N = 2, 3, 4` of the U(1) XXZ chain).
* `compression = true` additionally merges levels that carry only finish channels, which is lossy at order ``\tau^{N+1}`` (the error stays ``O(\tau^{N+1})``) but gives the smallest bonds. In the bulk of the Ising chain the bond dimensions with both options are `2, 3, 6, 12` for `N = 1, …, 4`, as in the paper. Symmetries and nilpotent letters can make them smaller still, e.g. `4, 13, 42, 130` for the U(1)-symmetric XXZ chain.
* Both options default to `extension = true, compression = false`.

Finite and infinite chains are both supported (on an `InfiniteChain`, `H` is a generating set as for [`mpo`](@ref OpSum.mpo)).
Only abelian symmetries are supported, fermions included; a non-abelian symmetry throws.
The total charge of `H` must be `unit`, and on a finite chain `H` may not contain [`expterm`](@ref OpSum.expterm) channels.
On-site products of letters are expanded in the site's basis, so the basis must be closed under products (e.g. `Basis(ops; complete = true)`); otherwise `project` throws that the product is not in the span of the basis.

With MPSKit loaded, `instantiate(U, τ)` gives an MPSKit `FiniteMPO` / `InfiniteMPO`, and `MPSKit.make_time_mpo(H, dt, TaylorCluster(; N, extension, compression); imaginary_evolution)` routes MPSKit's `TaylorCluster` to this symbolic builder.
Because ``\tau`` is symbolic, the MPO is built once and evaluated for every time step.

## Reading an error

| Message | Cause |
|---|---|
| `couple: no pair of terms of the operands fuses to …` | the requested total charge is unreachable |
| `couple: every term of the second operand must be a single-site charged operator` | the right operand has an identity part or spans several sites |
| `re-coupling the legs of a … coupling … is not supported` | out-of-order operands under a symmetry with multiplicities or non-symmetric braiding; write them in site order |
| `` `*` needs a forced channel `` | charged factors whose totals fuse several ways; name it with `couple(…; to = c)` |
| `… unresolved on-site product(s) …` | `length`/`≈` on a bag holding `a[i] * b[i]`; put it in an `OperatorSum` first |
| `mpo takes an OperatorSum` | a bare `Terms`; use `mpo(opsum(lat, h))` |
| `a term acts on site …, outside the lattice 1:N` | the term reaches past the `FiniteChain` |
| `a decaying channel …, so its letters land on sites with different bases` | `expterm` on mixed bases |
| `letter … does not exist on site …` | the operator was written for another basis |
| `all sites of a lattice must share one symmetry` / `cannot add an operator over J to an OperatorSum on a lattice of sector type I` | mixed symmetries |
| `a lattice is described by one operator Basis per site …` | a bare space as a site; use `FiniteChain(spin_basis(…), N)` |
| `cannot add an unplaced … to an OperatorSum` | place it first (`A[i]`, `B[i]`) |
| `adjoint: … no postfix h'` | `'` needs a lattice: `opsum(lat, h)'` |
| `project: unsupported operator space` | domain ≠ codomain, or a dual / degenerate charge leg |
| `project: the operator is not in the span of the basis` | use a complete basis or `Basis(ops; complete = true)` |
| `project: the projected term sum is not faithful` | `atol`/`rtol` discarded real weight; lower them |
| `GenericFusion … deferred` | fusion multiplicity > 1 is out of scope |
