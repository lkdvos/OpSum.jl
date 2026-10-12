# Building operators

This page is the guide to the operator interface: how to get a Hamiltonian into OpSum, and the
patterns that keep it correct.
The example pages — starting with [Spin chains](@ref) — work the same
material through concrete models; [Reference](@ref) lists every docstring.

## The shape of the pipeline

```
TensorMap ──project(O, B)──► SiteOperator / LocalOperator ──A[i], B[i, j]──► Terms ──opsum(lat, …)──► OperatorSum ──mpo──► SymbolicMPO
spin_ops, fermion_ops, … ───┘
```

A **site** is an operator [`Basis`](@ref OpSum.Basis): an ordered list of operators on the physical
space, the identity first, in which every on-site operator is expanded.
For the common sites there is a *stock basis* and a *builder* that hands out the operators already
expanded in it; for anything else you write a `TensorMap` and [`project`](@ref OpSum.project) it.

A bag of terms is **latticeless**: nothing in the term algebra needs a physical site, and the
compression itself needs only the *number* of sites.
The lattice enters when the terms are added to an [`OperatorSum`](@ref OpSum.OperatorSum), which
is the one container that holds a lattice (one basis per site), the terms placed on it, and any
exponentially decaying channels — and the one thing `mpo`, `instantiate` and `islossless` take.
Six types carry the whole interface:

| Type | What it is | How you get one |
|---|---|---|
| [`SiteOperator`](@ref OpSum.SiteOperator) | an operator on **one** site, not yet placed | a single-site entry of a builder (`spin_ops(U1Irrep).S_z`), [`project`](@ref OpSum.project)`(O, B)`, [`scalarop`](@ref OpSum.scalarop) |
| [`LocalOperator`](@ref OpSum.LocalOperator) | an operator on **`K` slots**, not yet placed | a two-site entry of a builder (`spin_ops(SU2Irrep).S_exchange`), [`project`](@ref OpSum.project)`(h, B)` |
| [`Terms`](@ref OpSum.Terms) | the compressible operator: a bag of [`Term`](@ref OpSum.Term)s, no lattice | `A[i]`, `B[i]` / `B[i, j]`, `*` of placed operators, [`opsum`](@ref OpSum.opsum), `+`, `append!`, `copy`, `zero`, `one`; the low-level [`couple`](@ref OpSum.couple) |
| [`FiniteChain`](@ref OpSum.FiniteChain) / [`InfiniteChain`](@ref OpSum.InfiniteChain) | the lattice: one basis per site | `FiniteChain(B, N)`, `FiniteChain([B₁, B₂, …])`, `InfiniteChain(B)`, `InfiniteChain([B₁, B₂])` |
| [`OperatorSum`](@ref OpSum.OperatorSum) | a lattice plus the terms (and channels) on it; letters are checked against the bases as they enter | [`opsum`](@ref OpSum.opsum)`(lat, terms...)`, `OperatorSum(lat)` + [`opsum!`](@ref OpSum.opsum!), `+`, `*`, `H'` |
| [`SymbolicMPO`](@ref OpSum.SymbolicMPO) | the compressed MPO, on either lattice; with MPSKit loaded `instantiate(W)` is MPSKit's Hamiltonian | [`mpo`](@ref OpSum.mpo) |

Everything named above is exported, so `using OpSum` is enough — you do not need a `using OpSum: …`
list.
The full surface is `Basis`, the stock bases `spin_basis`, `fermion_basis`, `boson_basis`,
`hubbard_basis`, `tj_basis` and their builders `spin_ops`, `fermion_ops`, `boson_ops`, `hubbard_ops`,
`tj_ops`, then `IrrepOperator`, `scalarop`, `project`, `LocalOperator`, `islossless`, `mpo_tensormap`,
`Term`, `Terms`, `couple`, `couple_channels`, `opsum`, `canonicalize!`, `OperatorSum`, `opsum!`,
`AbstractLattice`, `FiniteChain`, `InfiniteChain`, `mpo`, `SymbolicMPO`, `irrep_mpo_tensors`,
`jordan_mpo_tensors`, `mpo_terms`, `instantiate`, `expterm`, `ExpSum`, `BipartiteAlgorithm`,
`SVDBondAlgorithm` and the bond-basis strategies `VertexCover`, `IndependentSVD`, `SequentialSVD`.

Note that `dot` (for the Cartesian `Sᵢ·Sⱼ`) is `LinearAlgebra.dot`, so that one still needs
`using LinearAlgebra: dot`.

## Never name an alphabet letter by its index

Behind every operator is a set of letters [`IrrepOperator{I}(c, n)`](@ref
OpSum.IrrepTensorOperators.IrrepOperator): an operator charge `c` and an index `n`, the `n`-th
element of charge `c` in the site's basis.
A basis shows what it holds:

```@example ops
using OpSum, TensorKit

spin_basis(U1Irrep)
```

Four elements: the identity and `Sᶻ` (both of charge 0), and the two charged ones `S⁺` and `S⁻`.
Which element is which depends on the order the basis was built in, so **do not write those `n`s
into your code**: reorder the definition and every letter index permutes silently — you get a
different Hamiltonian and no error anywhere.

Write the operator down instead, and let OpSum find the letters.

## On-site operators

### Stock bases and builders

Each common site has a basis, `spin_basis`, `fermion_basis`, `boson_basis`, `hubbard_basis` and
`tj_basis`, taking the symmetry as an argument, and a builder of the same stem (`spin_ops`,
`fermion_ops`, `boson_ops`, `hubbard_ops`, `tj_ops`).
The builder returns a `NamedTuple` of every *charge-neutral* operator of that site that the symmetry
admits, expanded in the basis and named after the functions of
[TensorKitTensors.jl](https://github.com/QuantumKitHub/TensorKitTensors.jl):

```@example ops
B = spin_basis(U1Irrep)
ops = spin_ops(U1Irrep)
keys(ops)
```

A single-site entry is a `SiteOperator` (`ops.S_z`), placed with `ops.S_z[i]`; a two-site entry is an
unplaced [`LocalOperator`](@ref OpSum.LocalOperator) (`ops.S_plus_S_min`), placed with
`ops.S_plus_S_min[i, j]`.
The charged pieces — ``S^+`` and ``S^-``, a fermion's ``c^\dagger`` — are elements of the basis, but
you never write one down: they appear only inside the neutral operators that physics asks for.

```@example ops
length(ops.S_z)
```

`S_z` is a single element of the basis, one letter.
What the stock bases contain is chosen so that the operators one actually writes are atomic: a
Heisenberg term under SU(2) is one letter on each site, and for the Hubbard site
`hubbard_basis(U1Irrep, SU2Irrep)` the spinor ``c^\dagger_\sigma`` is a single element, so a hopping
term is one letter per site.
Each basis is also *complete*: it spans every operator on its space, so products and `H'` never
leave it.

```@example ops
Bsu2 = spin_basis(SU2Irrep)
S = spin_ops(SU2Irrep)
S.S_exchange       # S⃗ᵢ·S⃗ⱼ: one letter on each site, the vector operator
```

The symmetry decides what is *neutral*, and so what the builder offers: `spin_ops(SU2Irrep)` has only
the rotation-invariant `S_exchange`, `spin_ops(U1Irrep)` has `S_z`, `S_z_S_z`, `S_plus_S_min`,
`S_min_S_plus` and `S_exchange`, and `spin_ops()` (no symmetry) has the Cartesian components.
`spin_basis(sym; spin = 1)` and the like select the spin; `fermion_basis(U1Irrep)` has number
conservation, `fermion_basis()` only the parity (so that it also has the pairing `f_plus_f_plus`).
A fermionic mode is graded by `FermionParity ⊠ U1Irrep` or `FermionParity`, and the braiding is what
supplies the anticommutation signs.

### Anything else: `project`

For any other single-site operator, write the `TensorMap` and project it into the basis:

```@example ops
Bc = spin_basis()                  # no symmetry
Vc = space(Bc)
σx = project(TensorMap(ComplexF64[0 1; 1 0], Vc ← Vc), Bc)
```

Both a plain `V ← V` and a charged `V ← V ⊗ Vect[I](c => 1)` are accepted, and a plain matrix
(`project(Xmat, B)`) when there is no symmetry.
The expansion is exact: the coefficients are inner products, and `project` re-materializes its own
output and compares it against the input, throwing if the operator is outside the span of the basis
(the stock bases are complete, so it never is).

Nothing is ever converted to a dense `Array`, which matters for fermionic sectors where
`convert(Array, t)` is not well defined.

The identity is element 1 of every basis, and a scalar multiple of it is [`scalarop`](@ref
OpSum.scalarop); a placed identity is a pass-through, so it is dropped on placement and the term is
shorter.

### A basis of your own

`Basis(ops)` builds one from neutral operators, the identity implicit: a single-site operator is an
element as it is, and a two-site one is factorised into its charged pieces.
`complete = true` adds the matrix units it lacks, so that it spans everything.

```@example ops
Basis([ComplexF64[0 1; 1 0], ComplexF64[0 -im; im 0], ComplexF64[1 0; 0 -1]]; complete = true)
```

An operator is *atomic* in a basis when it is one of its elements, and atomic operators are what keep
the MPO small: the same Hamiltonian written in a poorer basis (for instance matrix units) has more
letters per term and a larger bond dimension, so put the operators that appear in your Hamiltonian
in the basis.

## Placing and multiplying

`A[i]` places an on-site operator on site `i` and gives a one-term [`Terms`](@ref OpSum.Terms) bag;
`B[i, j]` places a two-site `LocalOperator` on sites `i` and `j`; `B[i]` is contiguous (`i:i+K-1`).
The sites may be in **any order**, and need not be adjacent: the sites in between are
pass-throughs.

```@example ops
term = ops.S_plus_S_min[2, 5]
```

`*` between placed operators is the operator product, **in the order written**:

```@example ops
szz = ops.S_z[1] * ops.S_z[2]
szz ≈ ops.S_z_S_z[1, 2]
```

A few rules govern products and placement:

1. **Site order is free.**
   A term is *stored* in site order, so an operand acting to the left of an earlier one has its leg
   inserted rather than appended, and the braiding phase that costs comes with it.
   For fermions that phase is the anticommutation sign, so ``c^\dagger_{i+1} c_i`` is written as it
   reads (`f_plus_f_min[i + 1, i]`) and comes out as ``-c_i c^\dagger_{i+1}``.
   Under a non-abelian symmetry the product is formed in the order written and re-coupled into site
   order with F- and R-symbols (computed by TensorKit, cached per term shape), so one term written out of
   order can become several.
2. **Products on a repeated site are products in the site's basis.**
   `a[i] * b[i]` (or `hop[i, i + 1] * hop[i + 1, i + 2]`, which share a site) is resolved when it
   enters an `OperatorSum`, expressed in the basis of that site; a bag holding such a product is not
   in normal form until then, so `length` and `≈` on it throw.
   Because the stock bases are complete the answer is always in the basis.
3. **Fermionic products carry the graded sign.**
   Placing and multiplying two odd operators picks up the braiding of the legs that cross, which is the
   whole Jordan–Wigner bookkeeping, so there is no string to write.
4. **A product of charged operators needs a channel.**
   `*` needs the totals of the factors to fuse to one sector, which neutral operators — all the
   builders return — always do.

The products that physics asks for are products of the neutral operators of the builder.
A three-body term is a product of two bonds, and a four-body one a product of two:

```@example ops
ex = spin_ops(SU2Irrep).S_exchange
three = ex[1, 2] * ex[2, 3]          # (S₁·S₂)(S₂·S₃): the shared site is multiplied in the basis
four = ex[1, 3] * ex[2, 4]           # interleaved sites: re-coupled into site order
(typeof(three), typeof(four))
```

### Building blocks unplaced

A `LocalOperator` is the multi-site counterpart of a `SiteOperator`: the same term on `K` slots, placed
later with `B[i]` or `B[i, j]`.
Placement is a relabelling of the sites, so the projection (and its faithfulness check) is paid once
per *block*, not once per bond; `B[i, j]` is exact for any symmetry, and a fermionic hop across a gap
carries its string.
`LocalOperator`s add, subtract and scale, and `B + α` is a pass-through slot that placement drops:

```@example ops
b = spin_ops(SU2Irrep).S_exchange                         # K = 2, no site in sight
(b[2, 5] ≈ ex[2, 5], (b + 1 / 4)[1] ≈ ex[1, 2] + one(Terms{SU2Irrep}) / 4)
```

Placed and unplaced operands cannot be mixed in one call.

### The hermitian-conjugate partner

`H'` on an [`OperatorSum`](@ref OpSum.OperatorSum) builds it, so a hopping amplitude only has to be
written once:

```@example ops
Bf = fermion_basis(U1Irrep)
F = fermion_ops(U1Irrep)
flat = FiniteChain(Bf, 2)
T = opsum(flat, -1.0 * F.f_plus_f_min[1, 2])
T + T' ≈ opsum(flat, -1.0 * (F.f_plus_f_min[1, 2] + F.f_plus_f_min[2, 1]))
```

The adjoint of an alphabet letter is a combination of the dual charge's letters, which only the
basis knows, so `'` lives on the container and a bare `Terms` bag has none (it throws, pointing here).
It also works on an `InfiniteChain`.
Every term must be charge-neutral: a charged term's adjoint lives in the dual sector.
An operator holding `expterm` channels has no adjoint yet.

## Projecting a whole block

If you already have a ``K``-site operator as a `TensorMap` — from a paper, an ED code, a `kron` —
hand it to [`project`](@ref OpSum.project) together with the basis (or a list of bases, one per slot).
It need not factorize into ``A_i B_j``; a generic block works:

```@example ops
Bu = spin_basis(U1Irrep)
sz = ops.S_z_S_z
hbond = instantiate(
    opsum(
        FiniteChain(Bu, 2),
        (1 / 2) * ops.S_plus_S_min[1, 2] + (1 / 2) * ops.S_min_S_plus[1, 2] + sz[1, 2],
    )
)

Bxxz = project(hbond, Bu)
```

The result is a [`LocalOperator`](@ref OpSum.LocalOperator).
Indexing places it — `Bxxz[i]` on sites `i:i+K-1`, or `Bxxz[s₁, …, s_K]` on `K` explicitly named
sites — and gives back a `Terms` bag, so the projection (and its faithfulness check) is paid once per
*block*, not once per bond:

```@example ops
sites = FiniteChain(Bu, 6)
Hxxz = opsum(sites, Bxxz[i] for i in 1:5)
length(Hxxz)
```

Placement only relabels the sites, so `Bxxz[i, i + 2]` is exact for any symmetry (a fermionic hop
across a gap carries its string).
`project(h, [B₁, B₂, …])` expands slot `j` in `Bⱼ`, which is how a block between *different* sites
is written: the spin-½ / spin-1 exchange on the [Lattices and geometry](@ref) page is one.

Two accepted shapes, with `K = numout(h)`:

```
h : V_1 ⊗ … ⊗ V_K  ←  V_1 ⊗ … ⊗ V_K                        # charge-neutral
h : V_1 ⊗ … ⊗ V_K  ←  V_1 ⊗ … ⊗ V_K ⊗ Vect[I](tot => 1)    # total charge tot
```

The physical spaces are read off `h`.
The coefficients are exact inner products, so this is an expansion, not a fit — and `project`
re-materializes its own output and compares it against the input, throwing if the result is not
faithful.
`atol`/`rtol` control what counts as negligible.

!!! note "A projected block is a sum of terms"
    The identity is the first element of every basis, so an on-site identity factor of a projected
    block is a pass-through, which placement drops: ``\vec{S}_1 \cdot \vec{S}_2 + \tfrac{1}{4}``
    projects into a two-site term plus a constant, and the constant takes no site.

### Charged building blocks: `couple`

The neutral operators of the builders are enough for a Hamiltonian, and the charged elements of a basis
are reachable when you need them: `project(B[k], B)` is the one-letter operator for the `k`-th
element, a `TensorMap` of charge `c`, and [`couple`](@ref OpSum.couple) fuses placed charged operators
to a total charge, defaulting to the unit sector.
This is the low-level term algebra underneath `*`, and the one place the fusion channels are
visible:

```@example ops
Bs = spin_basis(SU2Irrep)
Sv = project(Bs[2], Bs)                     # the spin vector: the charge-1 element
couple_channels(Sv[1], Sv[2], Sv[3]; to = SU2Irrep(0))     # one tuple: nothing to name
```

```@example ops
chirality = couple(Sv[1], Sv[2], Sv[3])        # so the variadic form is unambiguous
```

* `couple(a, b; to = c)` fuses two operands to total charge `c`; `to` defaults to the unit sector, and
  an unreachable `to` is an error rather than a silently empty result.
* Every side may have several letters, and the coupling distributes over every combination; those
  whose charges cannot fuse to `to` are dropped.
* Three or more operands: the variadic form takes every channel the charges force, and errors only where
  a channel is a genuine choice (several singlet channels for four spin-1 operators).
  Nest `couple(couple(a, b; to = j₁₂), c; to = j)` to name it; `couple_channels` lists the legal
  tuples.
* Operands may be written in any site order; the legs are braided into place as for `*`.

## Building the MPO

```@example ops
W = mpo(Hxxz)
Ws, secs = W.Ws, W.bondsectors    # per site a sparse bond matrix, per bond its charges
map(length, secs)
```

`Ws[i]` is the sparse bond matrix at site `i` and `secs[i]` lists the charge of each bond index to
its right, so `length(secs[i])` is the reduced bond dimension and `sum(dim, secs[i])` the
dense-equivalent one.
`irrep_mpo_tensors(W)` assembles symmetric `TensorMap`s in the
MPSKit leg convention ``B_{i-1} \otimes V_i \leftarrow V_i \otimes B_i``.

The optional second argument selects the bond algorithm: `BipartiteAlgorithm()` (default, lossless
minimum vertex cover) or `SVDBondAlgorithm(trunc)` with a `MatrixAlgebraKit` truncation strategy.

### Where the lattice enters, and why there

The lattice enters with the [`OperatorSum`](@ref OpSum.OperatorSum), and that is not an oversight —
it is the first thing that needs one.
Three tiers:

| needs | what |
|---|---|
| site indices and charges only | `couple`, `dot`, placement, `Terms`' `opsum`, `canonicalize!`, `≈` |
| the site **count** `N` (or the cell period `L`) | the whole sweep: `mpo`, `mpo_terms` |
| the **spaces** | letter validation, `H'`, `instantiate`, `irrep_mpo_tensors` |

The compression never looks at a physical space: it reads `N` off the lattice and works on charges
and site indices alone.
The container is where the operator and the spaces are confronted: every letter is checked against
its site's space, and every term of a `FiniteChain` against `1:N`, **as the term is added**:

```@example ops
try
    opsum(FiniteChain(Bu, 3), Bxxz[i] for i in 1:5)       # a term reaches past site 3
catch e
    println(sprint(showerror, e))
end
```

A lattice may be a `FiniteChain`, an `InfiniteChain`, or any iterable naming one `Basis` per site (a
`Vector`, a tuple, a generator), which becomes a `FiniteChain`.
A lattice of bare spaces is an error: every site is a basis.

`opsum(lat, terms...)` builds the container in one pass, `opsum!(H, terms...)` fills one in place;
`H + x` copies `H`, so folding it over many terms is quadratic.
The natural shape for a model builder is to return the `OperatorSum`:

```julia
heisenberg(N; J = 1.0) = opsum(FiniteChain(Bsu2, N), J * S.S_exchange[i, i + 1] for i in 1:(N - 1))

mpo(heisenberg(64))
```

`FiniteChain(B, N)` is the uniform chain; `FiniteChain([B₁, B₂, …])` puts a different basis on every site.

`SVDBondAlgorithm` has two truncation semantics, chosen by its `sweep` keyword.
They agree exactly
when `trunc === nothing`, and differ only once truncation bites:

| `sweep` | meaning of `truncrank(k)` |
|---|---|
| `IndependentSVD` (default) | `k` retained indices **per bond** — every bond is compressed on the raw prefix/suffix classes, independently of its neighbours |
| `SequentialSVD` | `k` retained indices **after upstream truncation** — each bond is compressed in the basis the previous bond left behind, so an aggressive early truncation can starve the downstream bonds |

```julia
mpo(H, SVDBondAlgorithm(truncrank(8)))                        # per-bond
mpo(H, SVDBondAlgorithm(truncrank(8); sweep = SequentialSVD)) # left-to-right sweep
```

A basis need not be orthogonal (`n` overlaps `𝟙`), so a truncating SVD does not treat the letters as
orthonormal: it compresses in the Hilbert–Schmidt inner product of each site's basis and writes the
result back in that basis.
The truncated operator therefore depends on the span of the bases, not on how they were written.
Without truncation the basis does not matter, and nothing is changed.

### Jordan form, for an MPO library

`irrep_mpo_tensors` gives one dense `TensorMap` per site, with the whole bond collapsed into a single
`GradedSpace` — the right shape for contracting a small chain and checking it, the wrong one for an
MPO library, which wants the block structure back.
[`jordan_mpo_tensors`](@ref
OpSum.jordan_mpo_tensors) emits that instead: one `BlockTensorKit.SparseBlockTensorMap` per site,
one level per bond index, and the bond indices reordered into upper-triangular (Jordan) form

```
⎛ 1  C  D ⎞     row/column 1   = "nothing placed yet"
⎜ ·  A  B ⎟     row/column end = "everything placed"
⎝ ·  ·  1 ⎠
```

```@example ops
Wsj = jordan_mpo_tensors(Hxxz)
map(W -> size(W, 4), Wsj)
```

This is what MPSKit's `JordanMPOTensor` and `FiniteMPOHamiltonian` / `InfiniteMPOHamiltonian`
consume; with MPSKit loaded, `instantiate(mpo(H))` builds them (MPSKit is only a weak dependency).
Since every bond space is already known exactly, the consumer needs no
space-deduction fixed point.

Both identity channels are emitted at every internal bond, even where the compression did not spend a
bond index on them: a chain with no on-site fields has nothing starting to the right of the last
bonds, so the minimum vertex cover legitimately omits the start channel there, and nothing has
finished yet at the first bonds.
The emitted MPO is therefore minimal among *Jordan-form* MPOs and
can exceed the cover's minimum (`mpo(H).bondsectors`) by at most two indices per bond — in practice `+1` at
the first internal bond and `+1` at the last, `0` in the bulk.
The padded channels are pure identity
chains that cost no dense storage and cannot change the operator.

The total charge must be trivial (a Jordan MPO's right boundary is an identity), and a truncation
aggressive enough to empty a bond is rejected rather than silently emitted.

## Operators that are not Hamiltonians

Nothing in the compression assumes a Hamiltonian, so the things you *measure* compress too, and by
the same call.
A total magnetisation `opsum(lat, ops.S_z[i] for i in 1:N)` is a pure one-site sum
(`D = 2`), a two-point correlator `ops.S_z[1] * ops.S_z[4]` is a one-term MPO (`D = 2`), and a
string order parameter is one term with a run of active sites.

The one place the two paths genuinely diverge is a **charged** operator — total charge other than
`unit(I)`, such as a bare ``c^\dagger_2``, written as `project(t, B)[2]` for a `TensorMap` `t` with a charge leg.
[`mpo`](@ref OpSum.mpo) compresses it happily, and
[`irrep_mpo_tensors`](@ref OpSum.irrep_mpo_tensors) assembles it: the bond data carries the charge
and the last bond is simply not the unit sector.
[`jordan_mpo_tensors`](@ref OpSum.jordan_mpo_tensors) **refuses** it, for the reason in the paragraph
above — a Jordan MPO's right boundary is an identity, and a charged operator has no such boundary.
So
build charged objects and measure with them through `mpo` and `irrep_mpo_tensors`; the Jordan emission is for
Hamiltonians handed to an MPS algorithm.

## Infinite chains

Put the terms on an `InfiniteChain` and the same term algebra builds an infinite MPO with a
repeating unit cell of `L = length(chain)` sites:

```@example ops
chain = InfiniteChain(Bu)            # one site per unit cell
Hgen = opsum(
    chain, 0.5 * ops.S_plus_S_min[1, 2] + 0.5 * ops.S_min_S_plus[1, 2] + ops.S_z[1] * ops.S_z[2]
)
Hinf = mpo(Hgen)
```

The call has the same shape as the finite one: the operator is an
[`OperatorSum`](@ref OpSum.OperatorSum) — its terms exactly what placement and `*` return — and the lattice
says which chain to tile it over.
What comes back is a [`SymbolicMPO`](@ref OpSum.SymbolicMPO) on the infinite chain.

The argument is a **generating set**, not the Hamiltonian: the operator represented is
``\sum_{n \in \mathbb{Z}} \mathrm{translate}(H, nL)``, so each translation class must appear exactly
once.
Writing both `ops.S_z_S_z[1, 2]` and `ops.S_z_S_z[2, 3]` on a one-site cell would count the
same class twice and is rejected by `mpo`; on a two-site cell they are different bonds and both are needed.

```@example ops
Hinf.bondsectors, Hinf.start, Hinf.finish
```

It holds `L` bond matrices and `L` bond-charge lists, with bond `0` identified with bond `L`, exactly
like a finite one holds `N`.
It also
carries the two boundary vectors as bond indices: `start[j]` is the channel on which nothing has begun
and `finish[j]` the one on which everything has finished.
`irrep_mpo_tensors(Hinf)` assembles `L`
site tensors that tile — site 1's left virtual space is site `L`'s right virtual space.

!!! note
    Terms must be charge-neutral, and `SVDBondAlgorithm` is not available here: it compresses each bond
    independently against a vacuum-terminated layout, with no bond basis that closes on itself.
    Terms
    with no support (`K = 0`) are rejected too, since `Σ_n c·𝟙` does not converge.

## Exponentially decaying interactions

A term sum can only hold couplings of finite range.
[`expterm`](@ref OpSum.expterm) adds the one
infinite-range family an MPO represents exactly at fixed cost — a geometric decay, which is a scalar
`λ` on the diagonal of a bond channel:

```math
\sum_{i<j} \lambda^{\,j-i-1}\, A_i\, S_{i+1}\cdots S_{j-1}\, B_j ,\qquad 0 < |\lambda| < 1 .
```

You write one **representative term** and say where the stretched gap is; everything else — the
charges, the caterpillar fusion tree — comes from that term, exactly as for a finite-range coupling:

```@example ops
Hexp = opsum(
    chain,
    0.5 * ops.S_plus_S_min[1, 2] + 0.5 * ops.S_min_S_plus[1, 2],              # nearest neighbour
    expterm(ops.S_z_S_z[1, 2]; decay = 0.6),                                  # + geometric Sᶻ tail
)                                                # or a FiniteChain
Hgeo = mpo(Hexp)
Hgeo.bondsectors
```

The `expterm` channels live in the same [`OperatorSum`](@ref OpSum.OperatorSum) as the finite-range
terms, so all four finite/infinite × plain/decaying cases use `mpo(H)`.
The lattice fixes the translation period: on an
`L`-site cell the entry and the exit both step by `L` (and, as above, `H` is a generating set), while on
a [`FiniteChain`](@ref OpSum.FiniteChain) every site is a possible entry and the operator represented
is the geometric sum truncated to the chain — [`OpSum.chain_terms`](@ref)`(H)` writes it out.

* `decay` counts **per site** of the string, so the shortest translate carries `λ` to the power of the
  representative's own gap and every further period costs `λ^L`.
  `|λ| ≥ 1` is rejected.
* `exitsite` (default: the representative's last site) is the first site of the *exit block*.
  Sites
  before it form the entry block, so multi-site blocks on either end work:
  `expterm(ops.S_z_S_z[1, 2] * ops.S_z[4]; decay = λ, exitsite = 4)`.
* `string` is an on-site operator carried by the stretched gap (default: the identity), projected
  into the sites' basis like any other operator (`string = project(P, B)`).
  It must be
  charge-neutral — a charged string would make the running bond charge drift along the loop.
  A channel's letters land on sites it does not name — its translates move by one site on a finite
  chain, and a string fills every site of the gap — so those sites must all carry the same basis; on
  an `InfiniteChain` with a mixed cell, a channel without a string is fine.
* A **sum of exponentials** is a sum of `expterm`s.
  Channels with different `λ` are linearly
  independent, so each costs one bond index; channels sharing `(λ, string, exit)` merge into one.

!!! note
    A channel's period is part of its declaration, so writing the same interaction on a larger unit
    cell is not free: on an `L`-site cell an all-pairs geometric coupling needs `L²` channels and costs
    `L` bond channels instead of one.
    Write the smallest cell you can.
    Jordan blocks
    (polynomial × exponential) and power-law *fitting* are not supported; `research/infinite-mpo.md` §7
    is the design note.

## Recommended patterns

**Write the operator, not the index.**
Anything that mentions a bare `IrrepOperator(c, n)` in model
code is a latent bug.
Go through the stock builders (`spin_ops`, `fermion_ops`, `hubbard_ops`, …) or `project`.

**Build local operators wherever reads best.**
The stock bases and their builders (`spin_basis`, `spin_ops`, `fermion_ops`, …) are
memoised per symmetry, so calling them inside the term loop costs a dictionary lookup.

**Hand every term to `opsum` at once.**
It makes one pass over each argument, so building an
`M`-term Hamiltonian is ``\Theta(M)`` — an all-to-all model on 256 sites (32640 terms) assembles in a
few hundredths of a second. `opsum(lat, terms...)` does the same into an `OperatorSum`.
Any number of arguments is allowed, and iterables nest, so several
families of terms need no `vcat`:

```julia
ex = spin_ops(SU2Irrep).S_exchange
h = opsum(
    lat,
    (J1 * ex[i, j] for (i, j) in nn_bonds),
    (J2 * ex[i, j] for (i, j) in nnn_bonds),
)
```

`h + term` and `append!(h, terms)` also work on a bag, and `append!` is likewise one pass; on an
`OperatorSum` the in-place route is `opsum!(H, terms...)`.
`append!`/`opsum!` mutate, so start from a bag you own (`zero(h)`, `Terms{I}()`, `OperatorSum(lat)`) or a `copy(h)`.
But `+` *copies* the
accumulated list, so folding it — `for t in terms; H = H + t; end` — is quadratic.
That is fine for a
handful of terms and the wrong choice for thousands.

The normal form (coincident terms summed, cancelled ones dropped) is not taken on the way in: it
happens once, in [`canonicalize!`](@ref OpSum.canonicalize!), when the term set is first observed or
compressed.
So `length(H)` always reports the number of terms the operator *has*, never the number
appended.

**Project a bond block once, place it per bond.**
`project(h, B)` (like every two-site entry of a builder) is an unplaced `LocalOperator`, and `B[i]` is a
relabelling of its terms, so a translation-invariant chain is `opsum(lat, Bxxz[i] for i in 1:(N - 1))` with a
single projection (and a single faithfulness check) however long the chain.
Build the `TensorMap` once outside the loop, too.

**Order the sites of a 2D lattice so bonds stay short.**
An MPO lives on a chain.
Column-major
ordering ``(x, y) \mapsto (x-1)L_y + y`` keeps every bond of a cylinder within ``L_y`` sites, which
is what makes the bond dimension linear in the circumference and independent of the length.

**Pick the largest symmetry you can express, and put the operators you use in the basis.**
The same Heisenberg chain needs 3 reduced bond
indices under SU(2) and 5 under U(1).
The reduced number is what a DMRG sweep pays for.

## Verifying an operator

Three checks, in increasing cost:

```@example ops
# 1. Faithfulness — cheap, symbolic, valid at any N and for any sector including fermionic.
#    `islossless` does the reconstruct-and-compare in one call: `≈` on two term bags compares the
#    canonical term sets exactly and the coefficients approximately.
islossless(Hxxz)
```

```@example ops
# 2. Tensor assembly — contract the MPO with `mpo_tensormap` and compare against the dense oracle.
#    Exponential in N, but it stays inside TensorKit, so it is valid for fermions too.
mpo_tensormap(irrep_mpo_tensors(W)) ≈ instantiate(Hxxz)
```

3. **Spectrum** — the physics check, and the only honest route for fermions against an external
   reference.
   Diagonalize `instantiate(H)` block by block, repeating each eigenvalue
   `dim(c)` times so spectra are comparable across symmetry groups.
   `examples/common.jl` has a
   `spectrum` helper; a `hermiticity_error` check is the sharpest detector of a wrong fermionic sign.

Note that check 1 says nothing after a truncating `SVDBondAlgorithm` — it assumes lossless
compression.

## Reading an error

| Message | Cause |
|---|---|
| `couple: no pair of terms of the operands fuses to …` | the requested total charge is unreachable from the operands' charges |
| `couple: every term of the second operand must be a single-site charged operator` | the right operand has a scalar (identity) part, or spans several sites |
| `re-coupling the legs of a … coupling … is not supported` | operators written out of site order under a symmetry with fusion multiplicities or non-symmetric braiding; write them in increasing site order |
| `` `*` needs a forced channel `` | a product of charged operators whose totals can fuse in several ways; the neutral operators of the builders never hit this, and for charged ones name the channel with `couple(…; to = c)` |
| `… unresolved on-site product(s) …` | `length`, iteration or `≈` on a bag holding `a[i] * b[i]`; it is resolved in the site's basis when it enters an `OperatorSum` |
| `mpo takes an OperatorSum` | a bare term bag; write `mpo(opsum(lat, h))` |
| `a term acts on site …, outside the lattice 1:N` | raised when the term is added: it reaches past the `FiniteChain` of the `OperatorSum` |
| `a decaying channel …, so its letters land on sites with different bases` | an `expterm` on a finite chain of mixed bases, or one with a string on a mixed cell |
| `letter … does not exist on site …` | the basis at that site carries no operator of that charge and index — the operator was written for another basis |
| `all sites of a lattice must share one symmetry` | a mixed lattice of bases with different sector types |
| `cannot add an operator over J to an OperatorSum on a lattice of sector type I` | operator and lattice carry different symmetries |
| `a lattice must be a FiniteChain, an InfiniteChain, or an iterable of operator Basises …` | the lattice slot got something else: a bare space (`FiniteChain(V, N)`; every site is a `Basis` now, `FiniteChain(spin_basis(…), N)`) or an algorithm selector passed one argument too early |
| `cannot add an unplaced … to an OperatorSum` | a `SiteOperator`/`LocalOperator` was passed to `opsum`; place it first (`A[i]`, `B[i]`) |
| `adjoint: … no postfix h'` | a bare `Terms` bag has no `'`; put it on a lattice, `opsum(lat, h)'` |
| `project: unsupported operator space` | domain ≠ codomain, or a charge leg that is dual or has degeneracy > 1 |
| `project: the operator is not in the span of the basis` | the operator needs an element the basis lacks; use a complete basis (the stock bases are), or `Basis(ops; complete = true)` |
| `project: the projected term sum is not faithful` | `atol`/`rtol` discarded real weight — lower them |
| `GenericFusion … deferred` | sectors with fusion multiplicity > 1 are out of scope |
