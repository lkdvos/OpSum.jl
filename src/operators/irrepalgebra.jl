# Term-list symbolic algebra over ITOs
# =====================================
# The ITO global algebra is a *sum of sparse, sited terms*, not an expression tree. Each term is a
# set of active sites, one ITO letter per active site, coupled by a caterpillar fusion tree to a
# total charge, weighted by a reduced coefficient. `op[site]`, `+`, `scale`/`*`, and `couple`/`·` all
# produce or extend terms.
#
# Storage: one append-only struct of arrays
# -----------------------------------------
# A `TermList` stores its terms as *columns* of two `K × M` arrays, `sites` (ascending, zero-padded)
# and `keys::ITOKey` — exactly the layout the MPO sweep consumes (`ITOTermTable`, irreptermtable.jl).
# An `ITOKey` is `(letter, running bond charge, vertex label)`, and for the left-nested (caterpillar)
# coupling this algebra supports, the per-position running bond charges *are* the fusion tree: they
# are `bondcharges(tree)` and `vertexlabels(tree)`, and `_tree_from_bonds` inverts that. So the tree
# is derived, not stored, and `couple` becomes an append rather than a `TensorKit.join`.
#
# Every operation appends; nothing rebuilds. `+` concatenates, so folding it over a generator is
# linear rather than quadratic. Duplicate columns and cancelled coefficients are resolved **once**,
# lazily, by `canonical` (irreptermtable.jl) — previously this normal form was computed twice, first
# by a `Dictionary{TermKey, coeff}` and then again by the term table's own dedup.
#
# `H.terms` is the canonicalising view onto that store: a `Dictionary{TermKey, ComplexF64}` built on
# demand and memoised. It is the observable notion of "the terms of `H`" — `length`, `keys` and
# `pairs` all go through it, so no consumer ever sees a pre-dedup row count.
#
# Design notes:
# * Supported term arities: K = length(active sites) ∈ {0, 1, 2, …} — identity, single-site field,
#   and left-nested (caterpillar) coupling of arbitrarily many sites. On-site products/powers are
#   deferred; genuine multiplicity (`GenericFusion`, vertices > 1) is deferred.
# * Site labels are lattice indices (`Int`, `1:N`).
# * A `TermList` may carry the `lattice` it lives on (the physical spaces). Placement cannot know it
#   — `spin(V)[i]` sees one space and no `N` — so it is attached by [`onlattice`](@ref), which is
#   also where every letter is checked against the space it sits on. Once attached, `irrep_mpo(H)`,
#   `instantiate(H)` and `jordan_mpo_tensors(H)` need no second argument.
# * The scalar identity is materialized structurally as `c * id(V)`, never via `one(::Type{A})`.

using TensorKit
using TensorKit: Sector, ElementarySpace, FusionTree, fusiontrees, unit, dim, id, Nsymbol,
    Rsymbol, Vect, domain, permute, sectors, fuse, FusionStyle, UniqueFusion, BraidingStyle,
    SymmetricBraiding, insertrightunit
import TensorKit: sectortype
using LinearAlgebra: LinearAlgebra
using .IrrepTensorOperators: IrrepOperator

# exports are consolidated in `src/OpSum.jl`

# sector type of the ITO alphabet
sectortype(::Type{IrrepOperator{I}}) where {I} = I

# OnsiteOp instantiation over ITOs (single-site materialization)
# --------------------------------------------------------------
"""
    instantiate(O::OnsiteOp, V::ElementarySpace)

Materialize a single-site operator over the ITO alphabet as the coefficient-weighted sum of its
letters' tensors. The pass-through letter instantiates to `id(V)`, so a scalar `c·𝟙` comes out as
`c * id(V)` structurally.
"""
function instantiate(O::OnsiteOp{I}, V::ElementarySpace) where {I}
    isempty(O) && throw(ArgumentError("cannot instantiate an empty OnsiteOp without a space"))
    return sum(((l, c),) -> c * instantiate(l, V), pairs(O))
end

# Named accessors
# ---------------
const _SPIN_CACHE = Dict{ElementarySpace, Any}()

"""
    spin(V::ElementarySpace)

The SU(2) rank-1 vector operator on a single-sector SU(2) space `V`, normalized so `spin(V)[i] ·
spin(V)[j]` densifies to the Cartesian `Sˣ⊗Sˣ + Sʸ⊗Sʸ + Sᶻ⊗Sᶻ`. Concretely
`spin(V) = √(s(s+1)(2s+1)) · IrrepOperator(SU2Irrep(1), 1)` (the reduced matrix element).

Memoised per space, so calling it inside a term loop is a dictionary lookup. See also
[`spin_ops`](@ref) for the U(1)-graded `(Sp, Sm, Sz)` form.
"""
function spin(V::ElementarySpace)
    return _cached(_SPIN_CACHE, V) do
        _spin(V)
    end
end

function _spin(V::ElementarySpace)
    sectortype(V) === SU2Irrep ||
        throw(ArgumentError("spin(V) requires an SU(2)-graded space, got $(sectortype(V))"))
    length(collect(sectors(V))) == 1 ||
        throw(ArgumentError("spin(V) requires a single SU(2) irrep sector"))
    s = only(sectors(V)).j
    scale = sqrt(s * (s + 1) * (2s + 1))
    return scale * OnsiteOp(IrrepOperator{SU2Irrep}(SU2Irrep(1), 1))
end

"""
    scalarop(c::Number, V::ElementarySpace)
    scalarop(c::Number, ::Type{I<:Sector})

A scalar (identity) local operator `c·𝟙` over the ITO alphabet; instantiates structurally to
`c * id(V)` (the identity is not an alphabet letter).
"""
scalarop(c::Number, ::Type{I}) where {I} = OnsiteOp{I}(ComplexF64(c))
scalarop(c::Number, V::ElementarySpace) = scalarop(c, sectortype(V))

# Caterpillar-tree constructors
# ------------------------------
# The 4-arg `FusionTree{I}(uncoupled, coupled, isdual, innerlines)` form is used everywhere (the
# 3-arg form is an abelian-only shortcut that throws for `MultipleFusion`); vertices default to `1`.

# K = 0 identity (empty) tree.
_idtree(::Type{I}) where {I <: Sector} = FusionTree{I}((), unit(I), (), ())

# K = 1 single-leg tree (uncoupled == coupled == c).
_leaftree(c::I) where {I <: Sector} = FusionTree{I}((c,), c, (false,), ())

# Rebuild a caterpillar tree from its per-active-position running bond charges `bonds`
# (`[c₁, innerlines…, total]`, length K) and vertex labels `verts` (`[1, μ₂…μ_K]`, length K) — the
# inverse of `bondcharges`/`vertexlabels`. This is what makes the stored `ITOKey` columns and a
# `TermKey`'s explicit `FusionTree` two spellings of the same thing.
function _tree_from_bonds(charges::AbstractVector{I}, bonds::AbstractVector{I}, verts::AbstractVector{Int}) where {I <: Sector}
    K = length(charges)
    K == 0 && return _idtree(I)
    unc = ntuple(i -> charges[i], K)
    isd = ntuple(_ -> false, K)
    inner = ntuple(i -> bonds[i + 1], max(0, K - 2))   # innerlines = bonds[2 : K-1]
    vtx = ntuple(i -> verts[i + 1], max(0, K - 1))     # vertices   = verts[2 : K]
    return FusionTree{I}(unc, bonds[K], isd, inner, vtx)
end

# Term normal form (view type)
# ----------------------------
"""
    TermKey{I<:Sector, S}

A fusion-resolved term (hashable key): `sites` (sorted, unique active sites), `ops` (one ITO letter
per active site, aligned with `sites`), and `tree::FusionTree{I}` — the left-nested (caterpillar)
coupling tree over the operator charges, whose `coupled` sector is the term's total charge
(accessor [`total`](@ref)). `K = length(sites)`: 0 = identity, 1 = field, ≥ 2 = caterpillar coupling.
Restricted to multiplicity-free fusion (all tree vertices `== 1`); `GenericFusion` is deferred.

This is the *key type of the canonical view* [`TermSum`](@ref) presents through `H.terms`, not the
storage: a term list keeps its columns of `(site, ITOKey)` and derives the tree from the per-position
running bond charges (`_tree_from_bonds`).
"""
struct TermKey{I <: Sector, S}
    sites::Vector{S}
    ops::Vector{IrrepOperator{I}}
    tree::FusionTree{I}
    function TermKey{I, S}(
            sites::Vector{S}, ops::Vector{IrrepOperator{I}}, tree::FusionTree{I}
        ) where {I, S}
        @assert length(sites) == length(ops) == length(tree.uncoupled) "sites/ops/tree arity mismatch"
        @assert all(i -> ops[i].c == tree.uncoupled[i], eachindex(ops)) "tree uncoupled charges must match op charges"
        @assert all(isone, tree.vertices) "GenericFusion (multiplicity > 1) coupling is deferred"
        return new{I, S}(sites, ops, tree)
    end
end

"""
    total(k::TermKey)

The total (coupled) charge of a term: `k.tree.coupled`.
"""
total(k::TermKey) = k.tree.coupled

function Base.:(==)(x::TermKey{I, S}, y::TermKey{I, S}) where {I, S}
    return x.tree == y.tree && x.sites == y.sites && x.ops == y.ops
end
function Base.hash(x::TermKey, h::UInt)
    h = hash(:TermKey, hash(x.tree, h))
    for (s, o) in zip(x.sites, x.ops)
        h = hash(o, hash(s, h))
    end
    return h
end
function Base.show(io::IO, k::TermKey)
    return print(io, "TermKey(sites=", k.sites, ", ops=", k.ops, ", total=", total(k), ")")
end

# Append-only column store
# ------------------------
# THE ONE INVARIANT. A `TermBuffer`'s first `n` columns are immutable for the lifetime of the
# buffer: writes only ever happen *past* `n`, and `n` only ever grows. A `TermList` is therefore a
# perfectly ordinary immutable value even though several of them share one buffer — it is the
# buffer's first `list.n` columns, and nothing can overwrite them.
#
# That is what makes `+` amortized `O(#columns of the right operand)` rather than `O(total)`: when
# the left operand is the buffer's live tip (`a.n == a.buf.n`) the right operand is appended in
# place and the result is the new tip. Anything else — an older value being added to again, or a
# wider term arriving than the buffer's stride `K` can hold — forks a fresh buffer and copies, which
# costs `O(total)` but happens at most once per distinct stride along a fold (the stride only ever
# grows), so a left fold over `M` terms stays `Θ(M · maxarity)`.
mutable struct TermBuffer{I <: Sector}
    const K::Int                      # column stride: the max term arity this buffer can hold
    n::Int                            # columns written so far
    const sites::Vector{Int}          # column-major, K per column; 0 pads an inactive slot
    const keys::Vector{ITOKey{I}}
    const coeffs::Vector{ComplexF64}
end
TermBuffer{I}(K::Int) where {I <: Sector} =
    TermBuffer{I}(K, 0, Int[], ITOKey{I}[], ComplexF64[])

# the symbol an inactive (padded) slot carries, matching `_op_at_ito`'s reconstruction
_padkey(::Type{I}) where {I <: Sector} = ITOKey{I}(passthrough(I), unit(I), 1)

function _sizehint!(buf::TermBuffer, ncols::Int)
    sizehint!(buf.sites, ncols * buf.K)
    sizehint!(buf.keys, ncols * buf.K)
    sizehint!(buf.coeffs, ncols)
    return buf
end

"""
    TermSum{I<:Sector}
    TermList{I<:Sector}

The ITO global algebra: an append-only list of sited, fusion-coupled terms with reduced
coefficients, optionally bound to the lattice (`Vector{<:ElementarySpace}`) it lives on.

Built by `op[site]`, combined by `+`/`scale`/`*` and `couple`/`·`, bound to a lattice by
[`onlattice`](@ref), and consumed by [`irrep_mpo`](@ref) / [`instantiate`](@ref).

Duplicate terms and cancelled coefficients are resolved lazily, so `length(H)`, `keys(H)`,
`pairs(H)` and the `H.terms` dictionary view all report the *canonical* term set, never the number
of appended rows.
"""
mutable struct TermList{I <: Sector}
    const buf::TermBuffer{I}
    const n::Int                                          # this value owns columns 1:n of `buf`
    const lattice::Union{Nothing, Vector{<:ElementarySpace}}
    canon::Union{Nothing, TermList{I}}                    # memo: deduplicated, lattice-free
    dict::Union{Nothing, Dictionary{TermKey{I, Int}, ComplexF64}}   # memo: the `.terms` view
end

const TermSum = TermList

TermList(buf::TermBuffer{I}, n::Int = buf.n, lattice = nothing) where {I} =
    TermList{I}(buf, n, lattice, nothing, nothing)
TermList{I}() where {I <: Sector} = TermList(TermBuffer{I}(0))

# `H.terms` is a computed property; everything else is a field. Internal code goes through
# `getfield` so that the hot loops never depend on this being constant-folded.
function Base.getproperty(H::TermList, name::Symbol)
    name === :terms && return termdict(H)
    return getfield(H, name)
end
Base.propertynames(::TermList) = (:terms, :buf, :n, :lattice)

# Raw (pre-canonicalisation) column accessors. `t ∈ 1:nrows(H)`, `j ∈ 1:arity`.
nrows(H::TermList) = getfield(H, :n)
rowarity(H::TermList) = getfield(H, :buf).K
@inline colsite(H::TermList, j::Int, t::Int) =
    (b = getfield(H, :buf); @inbounds b.sites[(t - 1) * b.K + j])
@inline colkey(H::TermList, j::Int, t::Int) =
    (b = getfield(H, :buf); @inbounds b.keys[(t - 1) * b.K + j])
@inline colcoeff(H::TermList, t::Int) = @inbounds getfield(H, :buf).coeffs[t]

# number of active (non-padded) slots of column `t`
function collength(H::TermList, t::Int)
    K = rowarity(H)
    @inbounds for j in 1:K
        iszero(colsite(H, j, t)) && return j - 1
    end
    return K
end

# Copy columns `1:src.n` of `src` onto the end of `buf`, re-striding to `buf.K ≥ src` arity.
# Safe when `buf === getfield(src, :buf)`: only already-written slots are read.
function appendcols!(buf::TermBuffer{I}, src::TermList{I}) where {I}
    K, n = buf.K, nrows(src)
    Ks = rowarity(src)
    K >= Ks || throw(ArgumentError("term buffer stride $K cannot hold arity-$Ks columns"))
    pad = _padkey(I)
    for t in 1:n
        for j in 1:K
            if j <= Ks
                push!(buf.sites, colsite(src, j, t))
                push!(buf.keys, colkey(src, j, t))
            else
                push!(buf.sites, 0)
                push!(buf.keys, pad)
            end
        end
        push!(buf.coeffs, colcoeff(src, t))
    end
    buf.n += n
    return buf
end

# Append one column given its active `(site, key)` prefix.
function pushcol!(buf::TermBuffer{I}, sites, keys, coeff::ComplexF64) where {I}
    K, k = buf.K, length(sites)
    k <= K || throw(ArgumentError("term of arity $k does not fit a stride-$K buffer"))
    pad = _padkey(I)
    for j in 1:K
        push!(buf.sites, j <= k ? Int(sites[j]) : 0)
        push!(buf.keys, j <= k ? keys[j] : pad)
    end
    push!(buf.coeffs, coeff)
    buf.n += 1
    return buf
end

# Lattice
# -------
"""
    lattice(H::TermSum) -> Vector{<:ElementarySpace}

The physical spaces `H` is defined on. Throws if `H` is not bound to a lattice — see
[`onlattice`](@ref).
"""
function lattice(H::TermList)
    lat = getfield(H, :lattice)
    lat === nothing && throw(
        ArgumentError(
            "this operator is not bound to a lattice; call `onlattice(H, sites)` (or pass `sites` " *
                "to `irrep_mpo`/`instantiate`) to say which physical spaces it lives on"
        )
    )
    return lat
end

"""
    hascontext(H::TermSum) -> Bool

Whether `H` is bound to a lattice.
"""
hascontext(H::TermList) = getfield(H, :lattice) !== nothing

_tolattice(sites::Vector{<:ElementarySpace}) = sites
function _tolattice(sites)
    lat = collect(sites)
    eltype(lat) <: ElementarySpace || throw(
        ArgumentError("a lattice must be a vector of `ElementarySpace`s, got $(eltype(lat))")
    )
    return lat
end

# Per-space alphabet extent: `charge => number of letters of that charge on V`. Built once per
# *distinct* space of the lattice (compared by identity first, since `fill(V, N)` is the norm).
function _alphabets(lat::Vector{<:ElementarySpace}, ::Type{I}) where {I <: Sector}
    spaces = eltype(lat)[]
    infos = Dict{I, Int}[]
    which = Vector{Int}(undef, length(lat))
    for (i, V) in enumerate(lat)
        j = findfirst(W -> W === V || W == V, spaces)
        if j === nothing
            W = fuse(V ⊗ V')
            push!(spaces, V)
            push!(infos, Dict{I, Int}(c => dim(W, c) for c in sectors(W)))
            j = length(spaces)
        end
        which[i] = j
    end
    return infos, which
end

# Check that every stored letter can live on the space of its site. This is the only place the
# operators and the spaces they are compressed with are ever confronted; without it a `sites` vector
# from a different model silently produces a wrong MPO.
function _checklattice(H::TermList{I}, lat::Vector{<:ElementarySpace}) where {I}
    N = length(lat)
    isempty(lat) || sectortype(eltype(lat)) === I || throw(
        ArgumentError("lattice of sector type $(sectortype(eltype(lat))) for an operator over $I")
    )
    infos, which = _alphabets(lat, I)
    for t in 1:nrows(H), j in 1:rowarity(H)
        s = colsite(H, j, t)
        iszero(s) && break
        1 <= s <= N || throw(
            ArgumentError("a term acts on site $s, outside the lattice `1:$N`")
        )
        op = colkey(H, j, t).op
        info = infos[which[s]]
        nmax = get(info, op.c, 0)
        1 <= op.n <= nmax || throw(
            ArgumentError(
                "letter $op does not exist on site $s (space $(lat[s])): that space carries " *
                    "$nmax operator(s) of charge $(op.c)"
            )
        )
    end
    return nothing
end

"""
    onlattice(H::TermSum, sites) -> TermSum

Bind `H` to the lattice `sites` (a vector of physical spaces, one per site), checking that every
letter of every term exists on the space of the site it acts on and that no term reaches outside
`1:length(sites)`.

Placement cannot do this for you — `A[i]` sees one space and no system size — so a term list is
unbound until you say what it lives on. Once bound, [`irrep_mpo`](@ref), [`instantiate`](@ref) and
[`jordan_mpo_tensors`](@ref) take a single argument. Rebinding to a *different* lattice is an error.
"""
function onlattice(H::TermList{I}, sites) where {I}
    lat = _tolattice(sites)
    cur = getfield(H, :lattice)
    if cur !== nothing
        cur == lat && return H
        throw(
            ArgumentError(
                "this operator is already bound to a different lattice ($(length(cur)) sites); " *
                    "build it against the spaces you mean to compress it with"
            )
        )
    end
    _checklattice(H, lat)
    # the memos are lattice-free by construction, so they carry over unchanged
    return TermList{I}(
        getfield(H, :buf), getfield(H, :n), lat, getfield(H, :canon), getfield(H, :dict)
    )
end

function _mergelattice(a::TermList{I}, b::TermList{I}) where {I}
    la, lb = getfield(a, :lattice), getfield(b, :lattice)
    la === nothing && lb === nothing && return nothing
    if la !== nothing && lb !== nothing
        la == lb || throw(ArgumentError("cannot add operators defined on different lattices"))
        return la
    end
    lat = la === nothing ? lb : la
    _checklattice(la === nothing ? a : b, lat)
    return lat
end

# Arithmetic
# ----------
function Base.:+(a::TermList{I}, b::TermList{I}) where {I}
    lat = _mergelattice(a, b)
    buf = getfield(a, :buf)
    if buf.K >= rowarity(b) && getfield(a, :n) == buf.n
        # `a` is the live tip of its buffer: append in place (see THE ONE INVARIANT above)
        appendcols!(buf, b)
        return TermList(buf, buf.n, lat)
    end
    out = TermBuffer{I}(max(buf.K, rowarity(b)))
    _sizehint!(out, nrows(a) + nrows(b))
    appendcols!(out, a)
    appendcols!(out, b)
    return TermList(out, out.n, lat)
end

# One pass, one allocation: the whole point of the flat store. `Base.sum` on a vector otherwise
# reduces pairwise, which is `Θ(M log M)` copies rather than `Θ(M)`.
function Base.sum(terms::AbstractVector{<:TermList{I}}) where {I}
    isempty(terms) && return TermList{I}()
    K = maximum(rowarity, terms)
    lat = nothing
    for t in terms
        l = getfield(t, :lattice)
        l === nothing && continue
        lat === nothing ? (lat = l) :
            (lat == l || throw(ArgumentError("cannot add operators defined on different lattices")))
    end
    buf = TermBuffer{I}(K)
    _sizehint!(buf, sum(nrows, terms))
    for t in terms
        appendcols!(buf, t)
    end
    out = TermList(buf, buf.n, nothing)
    return lat === nothing ? out : onlattice(out, lat)
end

function VectorInterface.scale(a::TermList{I}, α::Number) where {I}
    buf = TermBuffer{I}(rowarity(a))
    _sizehint!(buf, nrows(a))
    appendcols!(buf, a)
    α′ = ComplexF64(α)
    buf.coeffs .*= α′
    return TermList(buf, buf.n, getfield(a, :lattice))
end
Base.:*(α::Number, a::TermList) = scale(a, α)
Base.:*(a::TermList, α::Number) = scale(a, α)
Base.:/(a::TermList, α::Number) = scale(a, inv(α))
Base.:-(a::TermList) = scale(a, -1)
Base.:-(a::TermList, b::TermList) = a + (-b)

function Base.one(a::TermList{I}) where {I}
    buf = TermBuffer{I}(0)
    pushcol!(buf, (), (), ComplexF64(1))
    return TermList(buf, buf.n, getfield(a, :lattice))
end

# Canonical view
# --------------
"""
    termdict(H::TermSum) -> Dictionary{TermKey, ComplexF64}

The canonical term dictionary of `H` — coincident terms summed, cancelled ones dropped — with each
term's caterpillar tree rebuilt from its stored running bond charges. Memoised; also reachable as
`H.terms`.
"""
function termdict(H::TermList{I}) where {I}
    d = getfield(H, :dict)
    d === nothing || return d
    c = canonical(H)
    if c !== H
        d = termdict(c)
        setfield!(H, :dict, d)
        return d
    end
    d = Dictionary{TermKey{I, Int}, ComplexF64}()
    for t in 1:nrows(H)
        insert!(d, colkey_term(H, t), colcoeff(H, t))
    end
    setfield!(H, :dict, d)
    return d
end

# Column `t` of a *canonical* list as a `TermKey`.
function colkey_term(H::TermList{I}, t::Int) where {I}
    k = collength(H, t)
    sites = Vector{Int}(undef, k)
    ops = Vector{IrrepOperator{I}}(undef, k)
    bonds = Vector{I}(undef, k)
    verts = Vector{Int}(undef, k)
    for j in 1:k
        key = colkey(H, j, t)
        sites[j] = colsite(H, j, t)
        ops[j] = key.op
        bonds[j] = key.bond
        verts[j] = key.vertex
    end
    charges = I[o.c for o in ops]
    return TermKey{I, Int}(sites, ops, _tree_from_bonds(charges, bonds, verts))
end

# Container interface — everything goes through the canonical form, so a pre-dedup row count is
# never observable. `length`/`isempty` stop at `canonical` and never build the fusion trees.
Base.length(H::TermList) = nrows(canonical(H))
Base.isempty(H::TermList) = iszero(nrows(canonical(H)))
Base.keys(H::TermList) = keys(termdict(H))
Base.values(H::TermList) = values(termdict(H))
Base.pairs(H::TermList) = pairs(termdict(H))
Base.getindex(H::TermList, k::TermKey) = termdict(H)[k]
Base.haskey(H::TermList, k::TermKey) = haskey(termdict(H), k)

"""
    isapprox(a::TermSum, b::TermSum; kwargs...)
    a ≈ b

Whether two term lists carry the same terms with matching coefficients: the canonical term sets must
be **equal** (a dropped term is never "approximately" absent) and the coefficients `≈`. Lattices are
not compared, so a term sum reconstructed by [`mpo_terms`](@ref) — which knows the bonds but not the
spaces — compares equal to the operator it came from. This is the faithfulness check.
"""
function Base.isapprox(a::TermList{I}, b::TermList{I}; kwargs...) where {I}
    da, db = termdict(a), termdict(b)
    length(da) == length(db) || return false
    for (k, v) in pairs(da)
        haskey(db, k) || return false
        isapprox(v, db[k]; kwargs...) || return false
    end
    return true
end

function Base.:(==)(a::TermList{I}, b::TermList{I}) where {I}
    return termdict(a) == termdict(b)
end

function Base.show(io::IO, ts::TermList)
    print(io, "TermSum(")
    join(io, ("$v * $k" for (k, v) in pairs(termdict(ts))), " + ")
    return print(io, ")")
end

# op[site]  →  a (possibly distributed) TermSum
# ---------------------------------------------
function Base.getindex(O::OnsiteOp{I}, ind::Integer, inds::Integer...) where {I}
    isempty(inds) ||
        throw(ArgumentError("multi-site placement of an on-site operator is not supported; use `couple`"))
    site = Int(ind)
    site >= 1 || throw(ArgumentError("site index must be ≥ 1, got $site"))
    # The pass-through letter is the bare identity: it carries no charge and contributes no active
    # site, so it places as a K=0 identity term rather than a K=1 field. A *real* trivial-charge
    # letter (`n ≥ 1`, e.g. from `project(id(V), V)`) is not the same thing and does place.
    K = any(!ispassthrough, keys(O)) ? 1 : 0
    buf = TermBuffer{I}(K)
    _sizehint!(buf, length(O))
    for (letter, coeff) in pairs(O)
        if ispassthrough(letter)
            pushcol!(buf, (), (), ComplexF64(coeff))
        else
            pushcol!(buf, (site,), (ITOKey{I}(letter, letter.c, 1),), ComplexF64(coeff))
        end
    end
    return TermList(buf)
end

# Coupling
# --------
# Second operand of `dot`: a single-site charged term; returns `(site, op, coeff)`.
function _single_site_term(b::TermList{I}, ctx) where {I}
    c = canonical(b)
    nrows(c) == 1 || throw(ArgumentError("$ctx must be a single-term operator"))
    collength(c, 1) == 1 ||
        throw(ArgumentError("$ctx must be a single-site charged operator"))
    return colsite(c, 1, 1), colkey(c, 1, 1).op, colcoeff(c, 1)
end

# Whether `b`'s leg may be *inserted* to the left of some of `a`'s, instead of only appended.
#
# `UniqueFusion` is what makes it well defined without F-moves, twice over: every running bond charge
# the insertion invalidates has a single forced replacement (so there is nothing for the caller to
# name), and every leg transposition is a single scalar `Rsymbol(x, y, x ⊗ y)` depending on nothing
# but the two charges. `SymmetricBraiding` (`Bosonic`/`Fermionic`, so `R = R⁻¹ = ±1`) is what makes
# the answer independent of whether the leg is braided over or under — a convention this API does not
# expose, and would have to before an anyonic sector could be reordered.
_canreorder(::Type{I}) where {I <: Sector} =
    FusionStyle(I) isa UniqueFusion && BraidingStyle(I) isa SymmetricBraiding

# Core of `couple`: extend every term of `a` by every single-site term of `b`, fusing to the total
# `target(running total of a's term, b's letter)` returns for that pair. Pairs whose charges cannot
# reach it are dropped, so the result may be empty — callers decide whether that is an error.
#
# Storage is always site-ordered. When `b` acts to the right of all of `a` — the left-nested
# caterpillar case — every earlier running bond charge is left alone and this is *literally* a column
# append: copy `a`'s active `(site, ITOKey)` slots, then one new slot
# `(site of b, ITOKey(letter of b, new total, vertex))`. Only the existence of the channel has to be
# checked, which is one `Nsymbol` — no fusion tree is built.
#
# When `b` acts further left, its leg is *inserted* at position `p` instead (see `_canreorder`): the
# running bond charges from `p` on are recomputed (forced, since fusion is unique) and the coefficient
# picks up one R-symbol per leg of `a` that `b` braids past. For fermionic sectors that product is
# exactly the anticommutation sign, so `couple(cd[j], c[i])` with `i < j` comes out as
# `-couple(c[i], cd[j])` without the caller supplying anything.
function _couple_terms(a::TermList{I}, b::TermList{I}, target) where {I}
    ca, cb = canonical(a), canonical(b)
    Ka = rowarity(ca)
    buf = TermBuffer{I}(Ka + 1)
    _sizehint!(buf, nrows(ca) * nrows(cb))
    reorder = _canreorder(I)
    sites = Vector{Int}(undef, Ka + 1)
    keys = Vector{ITOKey{I}}(undef, Ka + 1)
    for t in 1:nrows(ca)
        na = collength(ca, t)
        na >= 1 || throw(
            ArgumentError("couple: every term of the first operand must carry at least one charged operator")
        )
        ta = colkey(ca, na, t).bond
        va = colcoeff(ca, t)
        for u in 1:nrows(cb)
            collength(cb, u) == 1 || throw(
                ArgumentError("couple: every term of the second operand must be a single-site charged operator")
            )
            sb = colsite(cb, 1, u)
            opb = colkey(cb, 1, u).op

            # where `b`'s site falls among `a`'s: the first of them strictly to its right
            p = na + 1
            for j in 1:na
                s = colsite(ca, j, t)
                s == sb && throw(ArgumentError("couple: operators must act on distinct sites"))
                s > sb && (p = j; break)
            end
            (p > na || reorder) || throw(
                ArgumentError(
                    "couple: the second operand acts on site $sb, left of site " *
                        "$(colsite(ca, p, t)) of the first, and reordering the legs of a " *
                        "$(FusionStyle(I)) coupling needs F-moves, which is deferred. Write the " *
                        "operands in increasing site order."
                )
            )

            tot = target(ta, opb)::I
            nsym = Nsymbol(ta, opb.c, tot)
            iszero(nsym) && continue    # this pair of charges cannot fuse to `tot`: drop it
            @assert isone(nsym) "expected a unique coupling channel; multi-channel (GenericFusion) coupling is deferred"

            coeff = va * colcoeff(cb, u)
            for j in 1:(p - 1)
                sites[j] = colsite(ca, j, t)
                keys[j] = colkey(ca, j, t)
            end
            sites[p] = sb
            if p == na + 1
                keys[p] = ITOKey{I}(opb, tot, 1)
            else
                run = p == 1 ? opb.c : only(colkey(ca, p - 1, t).bond ⊗ opb.c)
                keys[p] = ITOKey{I}(opb, run, 1)
                for j in p:na
                    op = colkey(ca, j, t).op
                    coeff *= Rsymbol(op.c, opb.c, only(op.c ⊗ opb.c))
                    run = only(run ⊗ op.c)
                    sites[j + 1] = colsite(ca, j, t)
                    keys[j + 1] = ITOKey{I}(op, run, 1)
                end
                # unique fusion is associative *and* commutative, so the reordered chain lands on the
                # same total as the appended one would have; a mismatch means the recomputation drifted
                run == tot ||
                    _invariant("reordered coupling reached $run, not the total $tot")
            end
            pushcol!(buf, view(sites, 1:(na + 1)), view(keys, 1:(na + 1)), coeff)
        end
    end
    return TermList(buf, buf.n, _mergelattice(a, b))
end

"""
    couple(a::TermSum, b::TermSum; to = unit(I))
    couple(a::TermSum, b::TermSum, cs::TermSum...; to = unit(I))   # abelian only

Left-nested (caterpillar) irrep coupling: extend the composite `a` (any K ≥ 1 sites, with its running
coupling charges) by one single-site operator `b`, fusing the running total of `a` with `b`'s charge
to `to` at a new vertex. `to` defaults to the unit sector, which is what a term of a Hamiltonian
needs — pass it explicitly to build a charged object.

Under an **abelian** symmetry the operands may be written in any site order: the stored form is
site-ordered, so an out-of-order leg is inserted rather than appended, and the braiding phase that
costs is inserted with it. For a fermionic sector that phase *is* the anticommutation sign, so

```julia
couple(cd[i + 1], c[i]) == -couple(c[i], cd[i + 1])       # both spellings are available
```

and the hand-written sign that used to be mandatory is now only a way to get it wrong. Under a
non-abelian symmetry reordering needs F-moves and still throws: each operand after the first must
then act to the **right** of every site before it. ([`dot`](@ref) accepts either order for any
symmetry — with only two legs coupling to the unit sector, no F-move arises.)

Both operands may be composite (several terms, e.g. from [`project`](@ref)): the coupling
distributes over every pair of terms, and pairs whose charges cannot fuse to `to` are dropped. It is
an error if *no* pair fuses. Each term of `a` must carry at least one charged operator, and each
term of `b` must be a single charged site.

For ``K ≥ 3`` the intermediate channels matter. Under an **abelian** symmetry (`UniqueFusion`: `U₁`,
`ℤₙ`, `FermionNumber`, `Trivial`, products thereof) every intermediate is forced by the charges, so
the variadic form does the whole chain for you:

```julia
couple(cd[1], c[2], cd[3], c[4])        # a charge-neutral four-fermion term
```

Under a non-abelian symmetry the intermediates are genuine freedom and the variadic form throws:
nest instead, naming each channel, so the choice is explicit and readable back off the term:

```julia
couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))
```

This is the bare fusion coupler — it carries **no** normalization factor (reduced coeff `= va·vb`).
The Cartesian scalar-product convention lives in [`dot`](@ref), not here. Multi-channel
(`GenericFusion`) coupling and tree-structured (`via`) coupling are deferred.
"""
function couple(a::TermList{I}, b::TermList{I}; to = unit(I), via = nothing) where {I}
    via === nothing ||
        throw(ArgumentError("tree-structured / multi-body coupling (`via`) is deferred"))
    isempty(a) && throw(ArgumentError("couple: first operand has no terms"))
    isempty(b) && throw(ArgumentError("couple: second operand has no terms"))
    tot = to::I

    out = _couple_terms(a, b, (_, _) -> tot)
    isempty(out) &&
        throw(ArgumentError("couple: no pair of terms of the operands fuses to $tot"))
    return out
end

# Variadic coupling, abelian only: with `UniqueFusion` every intermediate charge is the single
# outcome of fusing the running total with the next operand's charge, so the whole caterpillar is
# fixed by the charges and there is nothing for the caller to choose. Fold left, letting each pair
# find its own intermediate, and constrain only the final total to `to`.
function couple(
        a::TermList{I}, b::TermList{I}, c::TermList{I}, rest::TermList{I}...;
        to = unit(I), via = nothing
    ) where {I}
    via === nothing ||
        throw(ArgumentError("tree-structured / multi-body coupling (`via`) is deferred"))
    FusionStyle(I) isa UniqueFusion || throw(
        ArgumentError(
            "couple: variadic coupling needs an abelian symmetry (unique fusion), but $I has " *
                "$(FusionStyle(I)) — the intermediate channels are a real choice there. Nest " *
                "`couple` to name each one, e.g. couple(couple(a, b; to = b₂), c; to = t)"
        )
    )
    isempty(a) && throw(ArgumentError("couple: first operand has no terms"))
    tot = to::I

    others = (b, c, rest...)
    acc = a
    for (i, nxt) in enumerate(others)
        isempty(nxt) &&
            throw(ArgumentError("couple: operand $(i + 1) has no terms"))
        # intermediates are forced; only the last step has to land on `to`
        acc = if i == length(others)
            _couple_terms(acc, nxt, (_, _) -> tot)
        else
            _couple_terms(acc, nxt, (ta, opb) -> only(ta ⊗ opb.c))
        end
        isempty(acc) && throw(
            ArgumentError(
                i == length(others) ?
                    "couple: no chain of the operands fuses to $tot" :
                    "couple: no pair of terms of operands 1..$(i + 1) can be coupled"
            )
        )
    end
    return acc
end

"""
    dot(a::TermSum, b::TermSum)
    a · b

The Cartesian two-body scalar product of two single-site ITO operators: singlet coupling
`couple(a, b; to = unit(I))` times the Cartesian factor `-√dim(c)` (the identity
`Sᵢ·Sⱼ = -√3 [S⊗S]⁽⁰⁾` with `-√3 = -√dim(spin-1)`).

Either site order is accepted, for **any** symmetry: two legs coupling to the unit sector is the one
case where reordering needs no F-move, whatever the fusion style, so `dot` is defined where the
general out-of-order [`couple`](@ref) is not. The Cartesian factor is read off the operator that ends
up on the left, which changes nothing — the two charges must fuse to the unit sector, i.e. be each
other's dual, and dual sectors have equal quantum dimension. The braiding phase does *not* always
cancel, though: `dot(a, b) == R · dot(b, a)` with `R = Rsymbol(cₐ, c_b, unit)`, which is `+1` for
bosonic integer charges (so `spin`, and hence every spin chain here, is order-independent) and `-1`
for a pair of odd fermionic charges or of half-integer SU(2) charges.

Unlike [`couple`](@ref) this does not distribute over composite operands: the Cartesian factor
`-√dim(c)` is per-letter, so it has no meaning for an operator mixing several charges. Use `couple`
for those.
"""
function LinearAlgebra.dot(a::TermList{I}, b::TermList{I}) where {I}
    sa, opa, _ = _single_site_term(a, "·: first operand")
    sb, opb, _ = _single_site_term(b, "·: second operand")
    sa == sb && throw(ArgumentError("·: operators must act on distinct sites"))
    iszero(Nsymbol(opa.c, opb.c, unit(I))) && throw(
        ArgumentError(
            "·: the two operator charges $(opa.c) and $(opb.c) do not fuse to the unit sector, so " *
                "there is no scalar product; use `couple(a, b; to = …)` for a charged coupling"
        )
    )
    sa < sb && return scale(couple(a, b; to = unit(I)), -sqrt(dim(opa.c)))

    # `b` sits to the left, so the two legs have to be swapped into storage order. With two legs
    # coupling to the unit sector that swap is one scalar R-symbol for any multiplicity-free fusion
    # style — no F-move, which is why this works under SU(2) where the general `couple` reordering
    # does not. Restricted to symmetric braiding so that `R = R⁻¹` and the (unexposed) over/under
    # convention cannot change the answer.
    BraidingStyle(I) isa SymmetricBraiding || throw(
        ArgumentError(
            "·: the operands are in decreasing site order and $I has $(BraidingStyle(I)) braiding, " *
                "where the swap phase depends on a braiding direction this API does not expose; " *
                "write the operands in increasing site order"
        )
    )
    R = Rsymbol(opa.c, opb.c, unit(I))
    return scale(couple(b, a; to = unit(I)), -sqrt(dim(opb.c)) * R)
end

# Dense-oracle materialization
# ----------------------------
"""
    instantiate(ts::TermSum[, sites::AbstractVector{<:ElementarySpace}])

Materialize the term-sum into a TensorKit `TensorMap` over the lattice `sites` (the dense oracle),
summing each term. `sites` defaults to `lattice(ts)` when `ts` is bound. Supports identity (K=0),
single-site field (K=1), and left-nested (caterpillar) coupling of any K ≥ 2 sites.
"""
function instantiate(ts::TermList{I}, sites::AbstractVector{<:ElementarySpace}) where {I}
    isempty(ts) && throw(ArgumentError("cannot instantiate an empty TermSum"))
    length(sites) == 0 && throw(ArgumentError("cannot instantiate over an empty lattice"))
    return sum(pairs(ts)) do (k, v)
        return v * _instantiate_term(k, sites)
    end
end
instantiate(ts::TermList) = instantiate(ts, lattice(ts))

function _instantiate_term(k::TermKey{I, S}, sites) where {I, S}
    N = length(sites)
    K = length(k.sites)
    if K == 0
        # The trailing total-charge leg is `Vect[I](unit(I) => 1)` for every K ≥ 1 term, so a K = 0
        # identity term needs one too — otherwise a term sum mixing the two cannot be summed at all,
        # which used to be a documented hole in the oracle.
        return insertrightunit(foldl(⊗, (id(sites[j]) for j in 1:N)))
    elseif K == 1
        return _embed_field(only(k.ops), only(k.sites), sites)
    else
        return _embed_caterpillar(k.ops, k.sites, k.tree, sites)
    end
end

# single charged field embedded on site `p`, identities elsewhere, charge leg to last domain slot
function _embed_field(op::IrrepOperator, p, sites)
    N = length(sites)
    loc = instantiate(op, sites[p])                # V_p ← V_p ⊗ V_c
    full = foldl(⊗, (j == p ? loc : id(sites[j]) for j in 1:N))
    cod = ntuple(identity, N)
    charge_global = N + (p + 1)
    dom_wo_charge = (ntuple(m -> N + m, p)..., ntuple(m -> N + p + 1 + m, N - p)...)
    dom = (dom_wo_charge..., charge_global)
    return permute(full, (cod, dom))
end

# Caterpillar-coupled K-site block (structural coupler; any reduced factor is in the coeff). The
# coupler `X` selects the *specific* fusion channel `tree` (setting its single reduced entry to 1 is
# a normalized isometry). Contraction of the K charge legs uses `permute` + morphism composition
# (bending the physical in-legs out and back), which generalizes to arbitrary K without a static
# `@tensor` expression.
function _embed_caterpillar(ops, positions, tree, sites)
    K = length(ops)
    I = typeof(tree.coupled)
    Os = [instantiate(ops[k], sites[positions[k]]) for k in 1:K]   # V ← V ⊗ Vc_k
    Vcs = [domain(Os[k])[2] for k in 1:K]
    tot = tree.coupled
    X = zeros(ComplexF64, foldl(⊗, Vcs) ← Vect[I](tot => 1))       # (Vc_1⊗…⊗Vc_K) ← Vect[tot]
    fcouple = FusionTree{I}((tot,), tot, (false,), ())
    X[tree, fcouple] .= 1

    P = foldl(⊗, Os)                                               # (o_1..o_K) ← (i_1,c_1,…,i_K,c_K)
    # bend physical in-legs into the codomain, leaving only the charge legs in the domain
    codP = (ntuple(k -> k, K)..., ntuple(k -> K + 2k - 1, K)...)   # o_1..o_K, i_1..i_K
    domP = ntuple(k -> K + 2k, K)                                  # c_1..c_K
    PX = permute(P, (codP, domP)) * X                              # (o.., i..) ← (tot)
    # split back into codomain (o_1..o_K) and domain (i_1..i_K, tot)
    Wblock = permute(PX, (ntuple(k -> k, K), (ntuple(k -> K + k, K)..., 2K + 1)))
    return _embed_block(Wblock, positions, sites)
end

# embed a K-site block on `positions` into the full lattice, reordering to site order with the
# total-charge leg last in the domain. `Wblock` codomain = (o over positions), domain = (i over
# positions, total-charge).
function _embed_block(Wblock, positions, sites)
    N = length(sites)
    K = length(positions)
    idle = [k for k in 1:N if !(k in positions)]
    full = isempty(idle) ? Wblock : Wblock ⊗ foldl(⊗, (id(sites[k]) for k in idle))

    cod_src = (positions..., idle...)
    p_cod = ntuple(j -> findfirst(==(j), cod_src), N)

    dom_src = (positions..., idle...)
    charge_global = N + (K + 1)                    # charge leg sits after the K active in-legs
    site_global(j) = let pos = findfirst(==(j), dom_src)
        pos <= K ? N + pos : N + pos + 1           # idle in-legs shift by 1 past the charge leg
    end
    p_dom = (ntuple(j -> site_global(j), N)..., charge_global)
    return permute(full, (p_cod, p_dom))
end
