# Building operators

This page is the guide to the operator interface: how to get a Hamiltonian into OpSum, and the
patterns that keep it correct. The example pages — starting with [Spin chains](@ref) — work the same
material through concrete models; [Reference](@ref) lists every docstring.

## The shape of the pipeline

```
TensorMap ──project──► SiteOperator ──A[i]──► TermSum ──irrep_mpo──► (Ws, bondsectors)
(what you                (on-site,    (placed and                    │
 write down)              unplaced)    coupled)                      ▼
                                          │              irrep_mpo_tensors
                                          └──instantiate──► TensorMap (dense oracle)
```

Three types carry the whole interface:

| Type | What it is | How you get one |
|---|---|---|
| [`SiteOperator`](@ref OpSum.SiteOperator) | an operator on **one** site, not yet placed | [`project`](@ref OpSum.project), [`matrixunit`](@ref OpSum.matrixunit), [`spin`](@ref OpSum.spin), [`scalarop`](@ref OpSum.scalarop) |
| [`TermSum`](@ref OpSum.TermSum) | a sum of sited, fusion-coupled terms — the Hamiltonian | `A[i]`, [`couple`](@ref OpSum.couple), `dot`, `+`, `*` |
| `(Ws, bondsectors)` | the reduced MPO | [`irrep_mpo`](@ref OpSum.irrep_mpo) |

Everything named above is exported, so `using OpSum` is enough — you do not need a `using OpSum: …`
list. The full surface is `IrrepOperator`, `spin`, `scalarop`, `project`, `matrixunit`, `TermSum`,
`couple`, `irrep_mpo`, `irrep_mpo_tensors`, `mpo_terms`, `instantiate`, `BipartiteAlgorithm` and
`SVDBondAlgorithm`.

Note that `dot` (for the Cartesian `Sᵢ·Sⱼ`) is `LinearAlgebra.dot`, so that one still needs
`using LinearAlgebra: dot`.

## Never name an alphabet letter by its index

The on-site alphabet is the set of irreducible tensor operators
[`IrrepOperator{I}(c, n)`](@ref OpSum.IrrepTensorOperators.IrrepOperator): an operator charge `c`
and a canonical index `n`. You can enumerate it:

```@example ops
using OpSum, TensorKit
using OpSum.IrrepTensorOperators: IrrepOperator

V = Rep[U₁](0 => 1, 1 => 1)
instances(IrrepOperator, V)
```

**Do not write those `n`s into your code.** `n` indexes TensorKit's canonical block ordering, which
depends on how `V` was spelled. Reorder the sectors of `V` and every letter index permutes silently —
you get a different Hamiltonian and no error anywhere.

Write the operator down instead, and let OpSum find the letters.

## On-site operators

### Abelian and fermionic: `matrixunit`

For spaces whose sectors are one-dimensional with multiplicity one — `Vect[U₁]`,
`Vect[FermionNumber]`, products thereof — name the matrix unit ``|out⟩⟨in|`` by its sectors:

```@example ops
dn, up = U1Irrep(0), U1Irrep(1)

Sp = matrixunit(V, up, dn)                                  # S⁺
Sm = matrixunit(V, dn, up)                                  # S⁻
Sz = (matrixunit(V, up, up) - matrixunit(V, dn, dn)) / 2     # Sᶻ
```

An `SiteOperator` supports ordinary arithmetic (`+ - * /`), so composite operators read the way you would
write them on paper. `Sᶻ` here is genuinely composite — two letters:

```@example ops
length(Sz)
```

### SU(2): `spin`

With SU(2) symmetry there is one on-site operator to speak of, the rank-1 vector operator
``\vec{S}``, normalized so that `dot` reproduces the Cartesian scalar product:

```@example ops
Vsu2 = SU2Space(1//2 => 1)
S = spin(Vsu2)
```

### Anything else: `project`

For any other single-site operator, build the `TensorMap` and project it. Both a plain `V ← V` and a
charged `V ← V ⊗ Vect[I](c => 1)` are accepted:

```@example ops
using LinearAlgebra: norm

Vf = Vect[FermionNumber](0 => 1, 1 => 1)
vac, occ = FermionNumber(0), FermionNumber(1)

nhat = matrixunit(Vf, occ, occ)                    # equivalently:
nhat2 = project(OpSum.instantiate(nhat, Vf), Vf)   # project the dense form back
norm(OpSum.instantiate(nhat, Vf) - OpSum.instantiate(nhat2, Vf))
```

Nothing is ever converted to a dense `Array`, which matters for fermionic sectors where
`convert(Array, t)` is not well defined.

The identity is *not* an alphabet letter. `project(id(V), V)` returns a sum of trivial-charge
letters, and a scalar multiple of the identity is [`scalarop`](@ref OpSum.scalarop).

## Placing and coupling

`A[i]` places an on-site operator on site `i`, distributing over its letters, and gives a one-term
`TermSum`. `couple(a, b; to = c)` fuses two of them into a two-site term with total charge `c`:

```@example ops
term = couple(Sz[1], Sz[2])
```

Four rules govern `couple`:

1. **Site order increases.** `couple` builds its caterpillar fusion tree left to right, so each
   operand must act strictly to the right of the ones before it. For fermions this is not a
   convention — writing ``c^†_{i+1} c_i`` as ``-c_i c^†_{i+1}`` costs a sign you must supply.
2. **Composite operands are fine.** Every side may have several letters; the coupling distributes
   over every combination, and combinations whose charges cannot fuse to `to` are dropped. It is an
   error if none fuse.
3. **`to` is the total charge of the term, and defaults to the unit sector.** A term in a
   Hamiltonian is a scalar, so the default is what you almost always want. Pass `to` explicitly to
   build a charged object — those are legal and useful as building blocks. If the charges cannot
   reach `to` you get an error rather than a silently empty result, which is what makes the default
   safe: `couple(Sp[1], Sp[2])` throws instead of quietly vanishing.
4. **Three or more sites: abelian folds, non-abelian nests.** Under an abelian symmetry
   (`UniqueFusion` — `U₁`, `ℤₙ`, `FermionNumber`, `Trivial`, products thereof) every intermediate
   charge is forced, so there is nothing to choose and the variadic form does the whole chain:

```@example ops
using OpSum: total

Vf = Vect[FermionNumber](0 => 1, 1 => 1)
cop = matrixunit(Vf, FermionNumber(0), FermionNumber(1))
cdag = matrixunit(Vf, FermionNumber(1), FermionNumber(0))

H4 = couple(cdag[1], cop[2], cdag[3], cop[4])       # charge-neutral four-fermion term
total(only(keys(H4.terms)))
```

Under a non-abelian symmetry the intermediates are real freedom, so the variadic form throws and you
nest, naming each channel — which also means the choice reads back off the term:

```@example ops
chirality = couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))
```

For the SU(2) scalar product ``\vec{S}_i \cdot \vec{S}_j`` use `dot`, which is
`couple(...; to = unit(I))` times the Cartesian factor ``-\sqrt{\dim c}``:

```@example ops
using LinearAlgebra: dot

heisenberg(N; J = 1.0) = sum([J * dot(S[i], S[i + 1]) for i in 1:(N - 1)])
H = heisenberg(6)
```

`dot` does **not** distribute over composite operands — its factor is per-letter, so it has no
meaning for an operator mixing charges. Use `couple` for those.

## Projecting a whole block

If you already have a ``K``-site operator as a `TensorMap` — from a paper, an ED code, a `kron` —
hand it to [`project`](@ref OpSum.project) directly. It need not factorize into ``A_i B_j``; a
generic block works:

```@example ops
hbond = OpSum.instantiate(
    (1 / 2) * couple(Sp[1], Sm[2]) +
        (1 / 2) * couple(Sm[1], Sp[2]) +
        couple(Sz[1], Sz[2]),
    [V, V],
)

Hxxz = sum([project(hbond, [i, i + 1]) for i in 1:5])
length(Hxxz.terms)
```

Two accepted shapes, with `K = length(sites) = numout(h)`:

```
h : V_1 ⊗ … ⊗ V_K  ←  V_1 ⊗ … ⊗ V_K                        # charge-neutral
h : V_1 ⊗ … ⊗ V_K  ←  V_1 ⊗ … ⊗ V_K ⊗ Vect[I](tot => 1)    # total charge tot
```

The physical spaces are read off `h`; `sites` supplies only the labels, and must be strictly
increasing. The coefficients are exact inner products against a complete orthogonal basis, so this
is an expansion, not a fit — and `project` re-materializes its own output and compares it against
the input, throwing if the result is not faithful. `atol`/`rtol` control what counts as negligible.

!!! warning "Projected terms have full support"
    Every returned term is active on **all** `K` sites. An on-site identity factor comes back as a
    trivial-charge letter, not as a shorter term. So `project` inverts `instantiate` only for
    operators whose terms all have full support on `sites` — projecting
    ``\vec{S}_1 \cdot \vec{S}_2 + \tfrac{1}{4}`` gives two two-site terms, not a two-site term plus
    a constant. The MPO is still correct; it may just carry a channel you would have written more
    compactly by hand.

## Building the MPO

```@example ops
using OpSum: irrep_mpo, irrep_mpo_tensors, mpo_terms

sites = fill(V, 6)
Ws, secs = irrep_mpo(Hxxz, sites)
map(length, secs)
```

`Ws[i]` is the sparse bond matrix at site `i` and `secs[i]` lists the charge of each bond index to
its right, so `length(secs[i])` is the reduced bond dimension and `sum(dim, secs[i])` the
dense-equivalent one. `irrep_mpo_tensors(Ws, secs, sites)` assembles symmetric `TensorMap`s in the
MPSKit leg convention ``B_{i-1} \otimes V_i \leftarrow V_i \otimes B_i``.

The third argument selects the bond algorithm: `BipartiteAlgorithm()` (default, lossless minimum
vertex cover) or `SVDBondAlgorithm(trunc)` with a `MatrixAlgebraKit` truncation strategy.

## Recommended patterns

**Write the operator, not the index.** Anything that mentions a bare `IrrepOperator(c, n)` in model
code is a latent bug. Go through `matrixunit`, `spin` or `project`.

**Hoist local operators out of the term loop.** `spin(V)` recomputes its normalization on every
call, and `matrixunit` runs a projection. Build them once.

**Collect terms into a `Vector`, then `sum`.** Never `reduce(+, generator)`: adding two `TermSum`s
rebuilds the underlying dictionary, so folding over a generator is quadratic in the number of terms.
`sum([...])` reduces pairwise and avoids that.

```julia
Sop = spin(Vphys)                                        # hoisted out of the loop
H = sum([J * dot(Sop[i], Sop[j]) for (i, j) in bonds])   # a Vector, not reduce(+, generator)
```

**Build a bond block once, project per bond.** `project` is cheap for local blocks (well under a
millisecond for a two-site spin-½ operator), so projecting the same tensor on each bond of a
translation-invariant chain costs nothing worth optimizing. Build the `TensorMap` once outside the
loop.

**Order the sites of a 2D lattice so bonds stay short.** An MPO lives on a chain. Column-major
ordering ``(x, y) \mapsto (x-1)L_y + y`` keeps every bond of a cylinder within ``L_y`` sites, which
is what makes the bond dimension linear in the circumference and independent of the length.

**Pick the largest symmetry you can express.** The same Heisenberg chain needs 3 reduced bond
indices under SU(2) and 6 under U(1). The reduced number is what a DMRG sweep pays for.

## Verifying an operator

Three checks, in increasing cost:

```@example ops
# 1. Faithfulness — cheap, symbolic, valid at any N and for any sector including fermionic.
back = mpo_terms(Ws, secs)
Set(keys(back.terms)) == Set(keys(Hxxz.terms)) &&
    all(back.terms[k] ≈ Hxxz.terms[k] for k in keys(Hxxz.terms))
```

```@example ops
# 2. Tensor assembly — contract the MPO and compare against the dense oracle. Exponential in N,
#    but it stays inside TensorKit, so it is valid for fermions too.
function mpo_tensormap(Ts)
    N = length(Ts)
    net = [[i == 1 ? -(2N + 1) : i - 1, -i, -(N + i), i == N ? -(2N + 2) : i] for i in 1:N]
    O = ncon(Ts, net)
    O = removeunit(O, 2N + 1)
    return permute(O, (ntuple(identity, N), (ntuple(i -> N + i, N)..., 2N + 1)))
end

mpo_tensormap(irrep_mpo_tensors(Ws, secs, sites)) ≈ OpSum.instantiate(Hxxz, sites)
```

3. **Spectrum** — the physics check, and the only honest route for fermions against an external
   reference. Diagonalize `OpSum.instantiate(H, sites)` block by block, repeating each eigenvalue
   `dim(c)` times so spectra are comparable across symmetry groups. `examples/common.jl` has a
   `spectrum` helper; a `hermiticity_error` check is the sharpest detector of a wrong fermionic sign.

Note that check 1 says nothing after a truncating `SVDBondAlgorithm` — it assumes lossless
compression.

## Reading an error

| Message | Cause |
|---|---|
| `couple: the second operand must act to the right of the first` | site indices not increasing; reorder and supply the fermionic sign yourself |
| `couple: no pair of terms of the operands fuses to …` | the requested total charge is unreachable from the operands' charges |
| `couple: every term of the second operand must be a single-site charged operator` | the right operand has a scalar (identity) part, or spans several sites |
| `project: sites must be strictly increasing` | sorted-unique labels are a downstream invariant; `project` will not permute `h` for you, since that needs braiding and flips fermionic signs |
| `project: unsupported operator space` | domain ≠ codomain, or a charge leg that is dual or has degeneracy > 1 |
| `project: the projected term sum is not faithful` | `atol`/`rtol` discarded real weight — lower them |
| `GenericFusion … deferred` | sectors with fusion multiplicity > 1 are out of scope |
