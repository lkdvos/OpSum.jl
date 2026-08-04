# Named operator sets, and memoisation of the ones that cost something
# ====================================================================
# Two three-line derivations were re-written by hand across the tests, the examples and the benchmark
# models: the `Sp`/`Sm`/`Sz` block for a U(1)-graded spin site, and the `(c, cd, n)` triple for a
# fermionic mode. Both are pure *convention* — which sector is "up", how the ladder normalisation
# goes — and a convention with nine copies is a convention with nine chances to differ.
#
# They live here rather than in irrepprojection.jl because they are built out of `matrixunit`, which
# needs `project`. `matrixunit` and `spin` are memoised (utility/memo.jl), so calling either of these
# inside a term loop costs a dictionary lookup rather than a fresh projection.

using TensorKit: ElementarySpace, Sector, sectortype

# exports are consolidated in `src/OpSum.jl`

"""
    spin_ops(V::ElementarySpace, sectors::AbstractVector) -> NamedTuple
    spin_ops(V::ElementarySpace, up, dn)

`(; Sp, Sm, Sz)` of a U(1)-graded spin-`s` site, where `sectors` lists the `2s+1` sectors of `V` in
**descending magnetic quantum number** (`m = s, s-1, …, -s`):

    Sᶻ = Σ_m m |m⟩⟨m|,    S⁺ = Σ_m √(s(s+1) - m(m+1)) |m+1⟩⟨m|,    S⁻ = (S⁺)ᵀ

so that `couple(Sz[i], Sz[j]) + (couple(Sp[i], Sm[j]) + couple(Sm[i], Sp[j])) / 2` is the Cartesian
`S⃗ᵢ · S⃗ⱼ` — the same operator `dot(spin(V)[i], spin(V)[j])` builds under SU(2).

The `m` values come from the *order* of `sectors`, never from the sector labels. A `Vect[U₁]` spin
site is just as often labelled by particle number as by `m` (`Rep[U₁](0 => 1, 1 => 1)` for spin-½ is
the common spelling), so inferring them would be a silent guess. The two-sector case takes its
sectors directly: `spin_ops(V, up, dn)`.

Each sector of `V` must be one-dimensional with multiplicity one — what [`matrixunit`](@ref) needs.

See also [`spin`](@ref) for the SU(2) irreducible-tensor form, and [`fermion_ops`](@ref).
"""
function spin_ops(V::ElementarySpace, sectors::AbstractVector{I}) where {I <: Sector}
    n = length(sectors)
    n >= 2 ||
        throw(ArgumentError("spin_ops: need at least two sectors to have a ladder, got $n"))
    allunique(sectors) ||
        throw(ArgumentError("spin_ops: the sectors must be distinct, got $sectors"))
    # `s` is read off *how many* sectors there are, so a partial list is not a smaller spin — it is
    # the wrong ladder normalisation on a subspace, silently. `matrixunit` catches a sector that is
    # absent from `V`; only this catches a sector of `V` that was left out.
    Set(sectors) == Set(TensorKit.sectors(V)) || throw(
        ArgumentError(
            "spin_ops: `sectors` must list every sector of $V exactly once, since the spin `s` " *
                "follows from how many there are; got $sectors"
        )
    )
    s = (n - 1) / 2
    m(k) = s - (k - 1)                                  # descending: s, s-1, …, -s
    ladder(k) = sqrt(s * (s + 1) - m(k + 1) * (m(k + 1) + 1))
    Sz = sum(
        m(k) * matrixunit(V, sectors[k], sectors[k]) for k in 1:n if !iszero(m(k))
    )
    Sp = sum(ladder(k) * matrixunit(V, sectors[k], sectors[k + 1]) for k in 1:(n - 1))
    Sm = sum(ladder(k) * matrixunit(V, sectors[k + 1], sectors[k]) for k in 1:(n - 1))
    return (; Sp, Sm, Sz)
end
spin_ops(V::ElementarySpace, up::I, dn::I) where {I <: Sector} = spin_ops(V, I[up, dn])

"""
    fermion_ops(V = Vect[FermionNumber](0 => 1, 1 => 1)) -> NamedTuple
    fermion_ops(V::ElementarySpace, vac, occ)

`(; c, cd, n)` of a spinless fermionic mode: annihilation `c`, creation `c†` and number `n̂`, as
matrix units between the empty sector `vac` and the occupied sector `occ` of `V`.

No Jordan–Wigner string is involved anywhere: the odd-parity charge flowing along the virtual bond
carries the anticommutation, which is why these are ordinary single-letter operators. Both site
orders of a hopping term are available — [`couple`](@ref) inserts the sign — so

```julia
F = fermion_ops()
H = -t * sum(couple(F.cd[i], F.c[i + 1]) + couple(F.cd[i + 1], F.c[i]) for i in 1:(N - 1))
```

The one-argument form infers `vac`/`occ` for `Vect[FermionNumber]` only; any other graded space has
to name them.

See also [`spin_ops`](@ref), [`matrixunit`](@ref).
"""
function fermion_ops(V::ElementarySpace, vac::I, occ::I) where {I <: Sector}
    return (;
        c = matrixunit(V, vac, occ),
        cd = matrixunit(V, occ, vac),
        n = matrixunit(V, occ, occ),
    )
end
function fermion_ops(V::ElementarySpace = Vect[FermionNumber](0 => 1, 1 => 1))
    I = sectortype(V)
    I === FermionNumber || throw(
        ArgumentError(
            "fermion_ops(V) reads the empty and occupied sectors off `FermionNumber` only, but $V " *
                "has sectors of type $I; name them explicitly as `fermion_ops(V, vac, occ)`"
        )
    )
    return fermion_ops(V, FermionNumber(0), FermionNumber(1))
end
