# Explicit on-site operator bases: a letter `(c, n)` names the `n`-th element of charge `c` of a
# `Basis`. The identity is always element 1, the letter the sweep uses for pass-throughs. Only
# `instantiate` interprets letters. Design note: `research/operator-bases.md` §3.

# Explicit on-site operator bases: a letter `(c, n)` names the `n`-th element of charge `c`; the
# identity is element 1 (the pass-through letter). Only `instantiate` interprets letters.

using TensorKit: AbstractTensorMap, ElementarySpace, numin, numout, insertrightunit, space,
    isomorphism, ComplexSpace, Trivial, dual, sectors
using LinearAlgebra: LinearAlgebra, Hermitian, eigvals, norm

"""
    Basis(ops::AbstractVector; atol = 0, rtol = 1e-12, complete = false)

An ordered basis of on-site operators on one space `V`: the alphabet that [`project`](@ref) expresses
operators in. Each entry of `ops` is a charge-neutral operator `V ← V` (or a plain matrix, without
symmetry), used as an element, or `V ⊗ V ← V ⊗ V`, factorised into its charged single-site pieces
(for `S⁺ ⊗ S⁻` the pieces `S⁺` and `S⁻`).

Element 1 is the identity; single-site entries follow in the order given, then the factorised pieces.
An entry parallel to an earlier element of its charge is dropped; the elements of one charge must be
linearly independent (checked) but need not be orthogonal. `project` throws for an operator outside
the span. With `complete = true` the basis is extended by matrix units to span every operator on `V`.
"""
struct Basis{I <: Sector, S <: ElementarySpace, T <: AbstractTensorMap}
    space::S
    elements::Vector{T}                        # basis order; `elements[1]` is the identity
    letters::Vector{IrrepOperator{I}}          # the letter of each element
    bycharge::Dictionary{I, Vector{Int}}       # element positions per charge, in index order
    pinv::Dictionary{I, Matrix{ComplexF64}}    # per charge: matrix-unit coefficients → basis ones
end

TensorKit.space(B::Basis) = B.space
TensorKit.sectortype(::Type{<:Basis{I}}) where {I} = I
Base.length(B::Basis) = length(B.elements)
Base.getindex(B::Basis, k::Integer) = B.elements[k]

function Base.show(io::IO, B::Basis)
    print(io, "Basis(", B.space, "; ", length(B), " elements: ")
    join(io, (string(c, " => ", length(ks)) for (c, ks) in pairs(B.bycharge)), ", ")
    return print(io, ")")
end

_space(B::Basis) = B.space
scalarop(c::Number, B::Basis) = scalarop(c, sectortype(B))

# `charge => number of letters`, what `_checklattice` validates letters against
_alphabet(B::Basis{I}, ::Type{I}) where {I} = Dict{I, Int}(c => length(ks) for (c, ks) in pairs(B.bycharge))

"""
    instantiate(op::IrrepOperator, B::Basis)
    instantiate(O::SiteOperator, B::Basis)

The element of `B` that the letter `op` names, `V ← V ⊗ Vect[I](c => 1)`, or the combination
`Σ coeff · element` of a symbolic on-site operator (whose letters must share one charge).
"""
function instantiate(op::IrrepOperator{I}, B::Basis{I}) where {I}
    ks = get(B.bycharge, op.c, nothing)
    (ks === nothing || !(1 <= op.n <= length(ks))) && throw(
        ArgumentError("letter $op does not exist in the basis $B")
    )
    return B.elements[ks[op.n]]
end

function instantiate(O::SiteOperator{I}, B::Basis{I}) where {I}
    isempty(O) && throw(ArgumentError("instantiate: the operator is zero, so it has no charge"))
    c = first(keys(O)).c
    all(l -> l.c == c, keys(O)) || throw(
        ArgumentError("instantiate: the letters of $O carry several charges")
    )
    return sum(coeff * instantiate(l, B) for (l, coeff) in pairs(O))
end

# Building a basis
# ----------------

# `elems` followed by every matrix unit on `V`.
_completed_elements(V, elems) = AbstractTensorMap[elems; [instantiate(mu, V) for mu in instances(IrrepOperator, V)]]

_identityelement(V) = isomorphism(ComplexF64, V ← V ⊗ Vect[sectortype(V)](unit(sectortype(V)) => 1))

function Basis(ops::AbstractVector; atol::Real = 0, rtol::Real = 1.0e-12, complete::Bool = false)
    isempty(ops) && throw(ArgumentError("Basis: needs at least one operator to fix the space"))
    tensors = map(ops) do t
        t isa AbstractMatrix || return t
        size(t, 1) == size(t, 2) || throw(ArgumentError("Basis: a matrix operator must be square, got $(size(t))"))
        V = ComplexSpace(size(t, 1))
        return TensorMap(ComplexF64.(t), V ← V)
    end
    V = codomain(first(tensors))[1]
    singles = AbstractTensorMap[]
    pairs_ = AbstractTensorMap[]
    for (k, t) in enumerate(tensors)
        all(==(V), codomain(t)) || throw(
            ArgumentError("Basis: operator $k acts on $(codomain(t)), not on copies of $V")
        )
        space(t) == (codomain(t) ← codomain(t)) || throw(
            ArgumentError(
                "Basis: operator $k must be a charge-neutral operator `V ← V` or `V ⊗ V ← V ⊗ V`, " *
                    "got $(space(t)); the charged elements are obtained by factorising two-site ones"
            )
        )
        if numout(t) == 1
            push!(singles, insertrightunit(t))
        elseif numout(t) == 2
            push!(pairs_, t)
        else
            throw(ArgumentError("Basis: operator $k acts on $(numout(t)) sites; only one or two are supported"))
        end
    end
    pieces = AbstractTensorMap[]
    for h in pairs_
        append!(pieces, _factorise_pair(h; atol, rtol))
    end
    B = _basis_from_elements(V, [_identityelement(V); singles; pieces]; rtol)
    complete || return B
    return _basis_from_elements(V, _completed_elements(V, B.elements); rtol, skipdependent = true)
end

# Charged elements (the first the identity) → `Basis`: group by charge, drop elements parallel to an
# earlier one, check independence, precompute the projector from matrix-unit coefficients. With
# `skipdependent` an element in the span of the earlier ones of its charge is dropped instead of throwing.
function _basis_from_elements(
        V::ElementarySpace, elems::AbstractVector; rtol::Real = 1.0e-12, skipdependent::Bool = false
    )
    I = sectortype(V)
    els = map(elems) do e
        x = zeros(ComplexF64, space(e))
        copy!(x, e)
        return x
    end
    first(els) ≈ _identityelement(V) ||
        _invariant("the first element of a basis must be the identity")
    kept = similar(els, 0)
    letters = IrrepOperator{I}[]
    bycharge = Dictionary{I, Vector{Int}}()
    spans = Dict{I, Matrix{ComplexF64}}()    # per charge: orthonormal span of the kept elements
    for e in els
        space(e) == (V ← V ⊗ domain(e)[2]) && numin(e) == 2 ||
            _invariant("a basis element must be `V ← V ⊗ Vect[I](c => 1)`, got $(space(e))")
        c = only(sectors(domain(e)[2]))
        ks = get!(bycharge, c, Int[])
        any(ks) do k
            return abs2(inner(kept[k], e)) >= (1 - rtol) * real(inner(kept[k], kept[k])) * real(inner(e, e))
        end && continue
        if skipdependent
            a = ComplexF64[inner(instantiate(mu, V), e) for mu in _chargeunits(V, c)]
            Q = get(spans, c, zeros(ComplexF64, length(a), 0))
            r = a - Q * (Q' * a)
            norm(r) > sqrt(rtol) * norm(a) || continue
            spans[c] = [Q r / norm(r)]
        end
        push!(kept, e)
        push!(ks, length(kept))
        push!(letters, IrrepOperator{I}(c, length(ks)))
    end
    filter!(!isempty, bycharge)    # a charge whose only candidates were dropped
    pinv = Dictionary{I, Matrix{ComplexF64}}()
    for (c, ks) in pairs(bycharge)
        mus = _chargeunits(V, c)
        # `A[m, k]`: coefficient of element `k` on matrix unit `m` (the matrix units are orthonormal)
        A = ComplexF64[inner(instantiate(mu, V), kept[k]) for mu in mus, k in ks]
        G = A' * A
        λ = eigvals(Hermitian(G))
        minimum(λ) > rtol * maximum(λ) || throw(
            ArgumentError(
                "Basis: the elements of charge $c are linearly dependent (Gram eigenvalues $λ); " *
                    "drop the redundant operator"
            )
        )
        P = G \ A'
        # round-off from the solve would otherwise become letters with ~1e-16 coefficients, each of
        # which costs the sweep a term
        P[abs.(P) .<= rtol * maximum(abs, P)] .= 0
        insert!(pinv, c, P)
    end
    return Basis{I, typeof(V), eltype(kept)}(V, kept, letters, bycharge, pinv)
end

# The matrix-unit letters of charge `c` on `V`.
_chargeunits(V, c) = filter(mu -> mu.c == c, instances(IrrepOperator, V))

# Rank factorisation of a neutral two-site operator into charged single-site pieces. Per charge `c` of
# the left factor, `M[a, b]` is the coefficient on the matrix units `eₐ ⊗ e_b` and `M = C R` with `C` the
# pivot columns of `M` and `R` its reduced row echelon form; the pieces are the columns of `C` and rows
# of `R`, each scaled so its first entry above `rtol` is 1 (`S⁺ ⊗ S⁻ + S⁻ ⊗ S⁺` gives `S⁺`, `S⁻`, not a
# mixture). `project` puts the scale back as the pair's coefficient.
function _factorise_pair(h::AbstractTensorMap; atol::Real = 0, rtol::Real = 1.0e-12)
    V = codomain(h)[1]
    I = sectortype(V)
    bych = Dictionary{I, Vector{Term{I}}}()
    for t in _project_units(h; atol).terms.terms
        push!(get!(bych, t.keys[1].op.c, Term{I}[]), t)
    end
    blocks = map(sort!(collect(keys(bych)))) do c
        ts = bych[c]
        L = sort!(unique!([t.keys[1].op for t in ts]))
        R = sort!(unique!([t.keys[2].op for t in ts]))
        M = zeros(ComplexF64, length(L), length(R))
        for t in ts
            M[searchsortedfirst(L, t.keys[1].op), searchsortedfirst(R, t.keys[2].op)] += t.coeff
        end
        return (L, R, M)
    end
    θ = max(float(atol), rtol * maximum(((_, _, M),) -> maximum(abs, M), blocks; init = 0.0))
    pieces = AbstractTensorMap[]
    for (L, R, M) in blocks
        C, E = _echelon_factors(M, θ)
        for k in axes(C, 2)
            l, r = _leadingone(C[:, k], rtol), _leadingone(E[k, :], rtol)
            push!(pieces, sum(l[m] * instantiate(L[m], V) for m in eachindex(L) if !iszero(l[m])))
            push!(pieces, sum(r[m] * instantiate(R[m], V) for m in eachindex(R) if !iszero(r[m])))
        end
    end
    return pieces
end

# `M = C E` with `C = M[:, pivots]` and `E` its reduced row echelon form; a column is a pivot when its
# residual against the earlier pivots exceeds `θ`.
function _echelon_factors(M::AbstractMatrix, θ::Real)
    Q = zeros(eltype(M), size(M, 1), 0)
    pivots = Int[]
    for j in axes(M, 2)
        r = M[:, j] - Q * (Q' * M[:, j])
        norm(r) > θ || continue
        push!(pivots, j)
        Q = [Q r / norm(r)]
    end
    C = M[:, pivots]
    return C, C \ M
end

# `v` scaled so its first entry above `rtol` of the largest is 1; entries below that are zeroed (round-off).
function _leadingone(v::AbstractVector, rtol::Real)
    vmax = maximum(abs, v)
    w = map(x -> abs(x) > rtol * vmax ? x : zero(x), v)
    return w / w[findfirst(!iszero, w)]
end

# Projection into a basis
# -----------------------

"""
    project(O, B::Basis; atol = 0, rtol = 100eps) -> SiteOperator     # O : V ← V, or a matrix
    project(h, B::Basis; atol = 0, rtol = 100eps) -> LocalOperator    # h : V^⊗K ← V^⊗K, K ≥ 2
    project(h, Bs::AbstractVector{<:Basis}; …)                        # slot j in Bs[j]

Express an operator in the basis `B`: a [`SiteOperator`](@ref) for one site, an unplaced
[`LocalOperator`](@ref) for `K` sites (placed with `h[i]` or `h[s₁, …, s_K]`). A slot a term does not
act on holds the identity, which placement drops. An operator outside the span of `B` throws; a trailing
total-charge leg is accepted as in [`project`](@ref)`(h)`.
"""
function project(O::AbstractMatrix, B::Basis; kwargs...)
    V = B.space
    sectortype(V) === Trivial || throw(
        ArgumentError("project: a plain matrix needs a basis without symmetry, got one on $V")
    )
    size(O) == (dim(V), dim(V)) || throw(
        ArgumentError("project: a $(size(O)) matrix does not act on $V")
    )
    return project(TensorMap(ComplexF64.(O), V ← V), B; kwargs...)
end

project(h::AbstractTensorMap, B::Basis; kwargs...) = project(h, fill(B, numout(h)); kwargs...)

function project(h::AbstractTensorMap, Bs::AbstractVector{<:Basis{I}}; kwargs...) where {I}
    out = _project_terms(h, Bs; kwargs...)
    K = length(Bs)
    K == 1 || return LocalOperator{I}(out, K)
    ts = canonicalize!(out).terms
    return SiteOperator{I}(IrrepOperator{I}[only(t.keys).op for t in ts], ComplexF64[t.coeff for t in ts])
end

# Matrix-unit terms re-expressed letter by letter in `Bs[j]`: `(c, m)` becomes `Σₙ P[n, m] (c, n)`. A
# term with a charge the basis lacks is dropped (the residual check in `_project_terms` catches it).
function _rebasis(mu::AbstractVector{Term{I}}, Bs) where {I}
    out = Term{I}[]
    for t in mu
        maps = map(eachindex(t.keys)) do j
            key = t.keys[j]
            P = get(Bs[j].pinv, key.op.c, nothing)
            P === nothing && return nothing
            col = P[:, key.op.n - IrrepTensorOperators._muoffset(key.op.c)]
            return [
                (ITOKey{I}(IrrepOperator{I}(key.op.c, n), key.bond, key.vertex), col[n])
                    for n in eachindex(col) if !iszero(col[n])
            ]
        end
        any(isnothing, maps) && continue
        for combo in Iterators.product(maps...)
            keys = ITOKey{I}[first(x) for x in combo]
            push!(out, Term{I}(t.sites, keys, t.coeff * prod(last, combo)))
        end
    end
    return Terms{I}(out)
end

# The terms of `h` on the slots `1:K`, slot `j` in `Bs[j]`; the residual decides whether `h` was in the span.
function _project_terms(
        h::AbstractTensorMap, Bs::AbstractVector{<:Basis{I}}; atol::Real = 0, rtol::Real = _default_rtol(h)
    ) where {I}
    K = numout(h)
    length(Bs) == K || throw(
        ArgumentError("project: got $(length(Bs)) bases for an operator on $K sites")
    )
    all(j -> codomain(h)[j] == Bs[j].space, 1:K) || throw(
        ArgumentError(
            "project: `h` acts on $(codomain(h)), not on the spaces of the bases " *
                "$(join((B.space for B in Bs), " ⊗ "))"
        )
    )
    hc, _ = _with_charge_leg(h, K, I)
    hnorm = norm(hc)
    θ = max(float(atol), float(rtol) * hnorm)
    mu = _project_units(h; atol = 0, rtol = 0).terms.terms
    summed = canonicalize!(_rebasis(mu, Bs)).terms
    # drop round-off: a coefficient whose term, at the scale of its elements, is below the tolerance
    weight(t) = prod(j -> norm(instantiate(t.keys[j].op, Bs[j])), 1:K)
    out = Terms{I}(filter(t -> abs(t.coeff) * weight(t) > θ, summed))

    resid = isempty(out.terms) ? hnorm : norm(hc - _instantiate_terms(out, Bs))
    slack = 16 * eps(real(float(scalartype(hc)))) * sqrt(max(1, length(mu))) * hnorm
    resid <= θ + slack || throw(
        ArgumentError(
            "project: the operator is not in the span of the basis (residual $resid, ‖h‖ = $hnorm); " *
                "extend the basis"
        )
    )
    return out
end
