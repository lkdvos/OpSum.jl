# Named operator sets, and memoisation of the ones that cost something
# ====================================================================
# Two three-line derivations were re-written by hand across the tests, the examples and the benchmark
# models: the `Sp`/`Sm`/`Sz` block for a U(1)-graded spin site, and the `(c, cd, n)` triple for a
# fermionic mode. Both are pure *convention* — which sector is "up", how the ladder normalisation
# goes — and a convention with nine copies is a convention with nine chances to differ.
#
# They live here rather than in irrepprojection.jl because they are built out of `matrixunit`, which
# needs `project`. `matrixunit` and `spin` are memoised (utility/memo.jl), so calling these inside a
# term loop costs a dictionary lookup rather than a projection.

using TensorKit: ElementarySpace, Sector, sectortype, removeunit, unit

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

# Hermitian conjugation
# =====================
# `hc(T)` is what turns a hopping amplitude into a hermitian Hamiltonian, and there is no cheap
# symbolic route to it. The adjoint of a term factorises over sites — `(f ⊗ g)† = f† ⊗ g†` in any
# dagger category, so no leg reversal is involved — but the *letter* adjoints do not stay in the
# alphabet position-by-position: `instantiate(letter, V)'` carries the dual charge and is generally a
# combination of that charge's letters, and re-expressing the caterpillar coupler over the dual
# charges is categorical data (B-symbols), not bookkeeping.
#
# So this goes the reliable way instead: materialise the term's `K`-site block through the tested
# forward map, take the adjoint there, and `project` it back. That is correct by construction —
# `project` re-materialises its own output and throws if it is not faithful — and the cost is one
# projection per *distinct* `(letters, tree, spaces)`, memoised, rather than one per term. It needs the
# physical spaces, so `H` must be lattice-bound or `sites` given.

const _HC_CACHE = Dict{Any, Any}()

# Re-label a term list built on the local lattice `1:K` onto `sites` (ascending, so the column order
# is preserved and nothing has to be re-sorted).
function _resite(ts::TermList{I}, sites::AbstractVector{Int}) where {I}
    K = rowarity(ts)
    buf = TermBuffer{I}(K)
    _sizehint!(buf, nrows(ts))
    for t in 1:nrows(ts)
        n = collength(ts, t)
        pushcol!(
            buf,
            Int[sites[colsite(ts, j, t)] for j in 1:n],
            ITOKey{I}[colkey(ts, j, t) for j in 1:n],
            colcoeff(ts, t)
        )
    end
    return TermList(buf)
end

"""
    hc(H::TermSum[, sites]) -> TermSum

The hermitian conjugate of `H`, so that `H + hc(H)` is hermitian and `hc(hc(H)) ≈ H`. `sites`
defaults to `lattice(H)`; the physical spaces are needed because the adjoint of an alphabet letter is
generally a *combination* of the dual charge's letters, which only the space knows.

```julia
F = fermion_ops()
T = -t * couple(F.cd[i], F.c[i + 1])
H = T + hc(T)                              # ≡ -t (c†ᵢcᵢ₊₁ + c†ᵢ₊₁cᵢ)
```

Every term must have total charge `unit(I)` — the case a Hamiltonian term is in. A charged term's
adjoint lives in the dual charge sector, which is a different object than this signature can return.

Each distinct term *shape* costs one projection, memoised, so conjugating a whole Hamiltonian is
`O(number of distinct shapes)` rather than `O(number of terms)`.
"""
function hc(H::TermList{I}, sites = lattice(H)) where {I}
    isempty(H) && return onlattice(TermList{I}(), sites)
    parts = TermList{I}[]
    for (k, v) in pairs(H)
        K = length(k.sites)
        if iszero(K)
            buf = TermBuffer{I}(0)
            pushcol!(buf, Int[], ITOKey{I}[], conj(v))
            push!(parts, TermList(buf))
            continue
        end
        total(k) == unit(I) || throw(
            ArgumentError(
                "hc: the term on sites $(k.sites) has total charge $(total(k)); the adjoint of a " *
                    "charged term carries the dual charge $(dual(total(k))), so it is not a term of " *
                    "the same operator. Conjugate charge-neutral terms only."
            )
        )
        Vs = [sites[s] for s in k.sites]
        local_hc = _cached(_HC_CACHE, (k.ops, k.tree, Vs)) do
            blk = _instantiate_term(TermKey{I, Int}(collect(1:K), k.ops, k.tree), Vs)
            return project(removeunit(blk, 2K + 1)', 1:K)
        end
        push!(parts, scale(_resite(local_hc, k.sites), conj(v)))
    end
    return onlattice(sum(parts), sites)
end
