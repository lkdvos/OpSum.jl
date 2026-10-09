# Explicit on-site operator bases: the letters `(c, n)` of a symbolic operator name elements of a
# user-chosen `Basis` instead of matrix units. Design note: `research/operator-bases.md` §3.
#
# A letter is `(charge, index)` with the index counted *within* the charge, as in the matrix-unit
# alphabet, and the identity is always element 1, i.e. the letter `(unit, 1)` — the same letter the
# sweep uses for pass-throughs (`passthrough`). Nothing downstream of `project` knows which alphabet
# a letter came from: the sweep only compares letters, and only `instantiate` interprets them.

using TensorKit: AbstractTensorMap, ElementarySpace, numin, numout, insertrightunit, space,
    isomorphism, ComplexSpace, Trivial, dual, sectors
using LinearAlgebra: LinearAlgebra, Hermitian, eigvals, norm

"""
    Basis(ops::AbstractVector; atol = 0, rtol = 1e-12)

An ordered basis of on-site operators on one space `V`, the alphabet that [`project`](@ref)
expresses operators in and that a symbolic operator's letters index into.

A basis is *defined* from charge-neutral operators: each entry of `ops` is

* a single-site operator `V ← V` (or a plain matrix, when there is no symmetry), used as an element
  as it is, or
* a two-site operator `V ⊗ V ← V ⊗ V`, factorised into its charged single-site pieces (for
  `S⁺ ⊗ S⁻` the pieces `S⁺` and `S⁻`), which become elements.

Element 1 is always the identity: it is prepended, and an entry parallel to it (or to any other
element of the same charge) is dropped. Single-site entries come first, in the order given, then the
factorised pieces. The elements of one charge must be linearly independent, which is checked; they
need not be orthogonal, and the basis need not span all operators on `V` — `project` throws for an
operator outside its span.

Internally each element is a `TensorMap` `V ← V ⊗ Vect[I](c => 1)` of one definite charge `c`, and
the elements are grouped by charge: an element's letter is `(c, n)`, the `n`-th element of charge `c`
in basis order. A two-site operator is factorised canonically (column-pivoted reduced row echelon
form of its matrix-unit coefficients, see `_factorise_pair`), so a pair of single matrix units gives
those matrix units; each piece's first coefficient is 1, and the pair's scale shows up as its
coefficient when it is projected into the basis.

```julia
Imat = [1 0; 0 1]; Xmat = [0 1; 1 0]; Ymat = [0 -im; im 0]; Zmat = [1 0; 0 -1]
B = Basis([Imat, Xmat, Ymat, Zmat])          # four elements, the identity first
X = project(Xmat, B)                         # one letter: X is atomic in B
```
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
Base.iterate(B::Basis, s...) = iterate(B.elements, s...)
Base.eltype(::Type{<:Basis{I, S, T}}) where {I, S, T} = T

function Base.show(io::IO, B::Basis)
    print(io, "Basis(", B.space, "; ", length(B), " elements: ")
    join(io, (string(c, " => ", length(ks)) for (c, ks) in pairs(B.bycharge)), ", ")
    return print(io, ")")
end

_space(B::Basis) = B.space

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

function Basis(ops::AbstractVector; atol::Real = 0, rtol::Real = 1.0e-12)
    isempty(ops) && throw(ArgumentError("Basis: needs at least one operator to fix the space"))
    tensors = map(_basis_tensor, ops)
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
    I = sectortype(V)
    return _basis_from_elements(V, [isomorphism(ComplexF64, V ← V ⊗ Vect[I](unit(I) => 1)); singles; pieces]; rtol)
end

_basis_tensor(t::AbstractTensorMap) = t
function _basis_tensor(M::AbstractMatrix)
    size(M, 1) == size(M, 2) || throw(ArgumentError("Basis: a matrix operator must be square, got $(size(M))"))
    V = ComplexSpace(size(M, 1))
    return TensorMap(ComplexF64.(M), V ← V)
end

# Charged elements → `Basis`: group by charge, drop elements parallel to an earlier one of the same
# charge, check independence, and precompute the projector from matrix-unit coefficients. The first
# element must be the identity. Internal: the public constructor only takes neutral operators.
function _basis_from_elements(V::ElementarySpace, elems::AbstractVector; rtol::Real = 1.0e-12)
    I = sectortype(V)
    els = map(elems) do e
        x = zeros(ComplexF64, space(e))
        copy!(x, e)
        return x
    end
    first(els) ≈ isomorphism(ComplexF64, V ← V ⊗ Vect[I](unit(I) => 1)) ||
        _invariant("the first element of a basis must be the identity")
    kept = similar(els, 0)
    letters = IrrepOperator{I}[]
    bycharge = Dictionary{I, Vector{Int}}()
    for e in els
        c = _elementcharge(e, V)
        ks = get!(bycharge, c, Int[])
        any(k -> _isparallel(kept[k], e, rtol), ks) && continue
        push!(kept, e)
        push!(ks, length(kept))
        push!(letters, IrrepOperator{I}(c, length(ks)))
    end
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
        insert!(pinv, c, G \ A')
    end
    return Basis{I, typeof(V), eltype(kept)}(V, kept, letters, bycharge, pinv)
end

function _elementcharge(e::AbstractTensorMap, V)
    space(e) == (V ← V ⊗ domain(e)[2]) && numin(e) == 2 ||
        _invariant("a basis element must be `V ← V ⊗ Vect[I](c => 1)`, got $(space(e))")
    return only(sectors(domain(e)[2]))
end

_isparallel(a, b, rtol) = abs2(inner(a, b)) >= (1 - rtol) * real(inner(a, a)) * real(inner(b, b))

# The matrix-unit letters of charge `c` on `V`, in index order.
_chargeunits(V, c) = filter(mu -> mu.c == c, instances(IrrepOperator, V))

# The rank factorisation of a neutral two-site operator into charged single-site pieces. Per charge
# `c` of the left factor, `M[a, b]` is the pair's coefficient on the matrix units `eₐ ⊗ e_b`, and
# `M = C R` with `C` the pivot columns of `M` (the first columns, in letter order, that are independent
# of the ones before) and `R` the reduced row echelon form of `M`. The pieces are the columns of `C` on
# the left and the rows of `R` on the right, each scaled so that its first entry above `rtol` is 1.
#
# This is canonical — no singular vectors, so no rotation within a degenerate singular value — and
# readable: a pair built from single matrix units gives exactly those (`S⁺ ⊗ S⁻ + S⁻ ⊗ S⁺` gives `S⁺`
# and `S⁻`, not a mixture of them). Only the directions matter: `project` puts the scale back as the
# pair's coefficient.
function _factorise_pair(h::AbstractTensorMap; atol::Real = 0, rtol::Real = 1.0e-12)
    V = codomain(h)[1]
    I = sectortype(V)
    bych = Dictionary{I, Vector{Term{I}}}()
    for t in project(h; atol).terms.terms
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

# `M = C E` with `C = M[:, pivots]` and `E` the reduced row echelon form of `M` (unit vectors in the
# pivot columns). A column is a pivot when its residual against the earlier pivots exceeds `θ`.
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

# `v` scaled so that its first entry above `rtol` of its largest is exactly 1, with entries below
# `rtol` of the largest set to zero (round-off from the solve, which would otherwise show up as letters).
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

Express an operator in the basis `B`: a single-site operator gives a [`SiteOperator`](@ref), a
coefficient per element; a `K`-site one gives an unplaced [`LocalOperator`](@ref) whose every slot is
expressed in `B`, placed with `h[i]` or `h[s₁, …, s_K]`. A slot that a term does not act on holds the
identity (element 1), which placement drops, so a projected `X ⊗ 𝟙` places as a one-site term.

The elements need not be orthogonal: the coefficients solve against the basis per charge. An
operator outside the span of `B` throws, with the residual; the same tolerance drops coefficients
that are round-off. A trailing total-charge leg is accepted as in [`project`](@ref)`(h)`.
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

function project(h::AbstractTensorMap, B::Basis{I}; atol::Real = 0, rtol::Real = _default_rtol(h)) where {I}
    K = numout(h)
    all(==(B.space), codomain(h)) || throw(
        ArgumentError("project: `h` acts on $(codomain(h)), not on copies of the basis space $(B.space)")
    )
    hc, _ = _with_charge_leg(h, K, I)
    hnorm = norm(hc)
    θ = max(float(atol), float(rtol) * hnorm)
    # the matrix-unit expansion is exact, so this only changes coordinates
    mu = project(h; atol = 0, rtol = 0).terms.terms
    acc = Dictionary{Vector{ITOKey{I}}, ComplexF64}()
    slotmaps = map(mu) do t
        map(t.keys) do key
            P = get(B.pinv, key.op.c, nothing)
            P === nothing && return nothing
            ks = B.bycharge[key.op.c]
            m = key.op.n - IrrepTensorOperators._muoffset(key.op.c)
            col = P[:, m]
            return [(ITOKey{I}(IrrepOperator{I}(key.op.c, n), key.bond, key.vertex), col[n]) for n in eachindex(ks) if !iszero(col[n])]
        end
    end
    for (t, maps) in zip(mu, slotmaps)
        any(isnothing, maps) && continue                      # a charge the basis has no element of
        for combo in Iterators.product(maps...)
            keys = ITOKey{I}[first(x) for x in combo]
            c = t.coeff * prod(last, combo)
            set!(acc, keys, get(acc, keys, zero(ComplexF64)) + c)
        end
    end
    # drop round-off: a coefficient whose term, at the scale of its elements, is below the tolerance
    elnorm(key) = norm(instantiate(key.op, B))
    slots = collect(1:K)
    terms = Term{I}[Term{I}(slots, k, c) for (k, c) in pairs(acc) if abs(c) * prod(elnorm, k) > θ]
    out = Terms{I}(terms)

    Bs = fill(B, K)
    resid = isempty(terms) ? hnorm : norm(hc - _instantiate_terms(out, Bs))
    slack = 16 * eps(real(float(scalartype(hc)))) * sqrt(max(1, length(mu))) * hnorm
    resid <= θ + slack || throw(
        ArgumentError(
            "project: the operator is not in the span of the basis (residual $resid, ‖h‖ = $hnorm); " *
                "extend the basis"
        )
    )
    K == 1 || return LocalOperator{I}(out, K)
    isempty(terms) && return zero(SiteOperator{I})
    ts = canonicalize!(out).terms
    return SiteOperator{I}(IrrepOperator{I}[only(t.keys).op for t in ts], ComplexF64[t.coeff for t in ts])
end
