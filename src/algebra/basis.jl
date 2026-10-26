# Explicit on-site operator bases: a letter `(c, n)` names the `n`-th element of charge `c` of a
# `Basis`; the identity is element 1 (the pass-through letter). A `Basis` is always complete, so every
# operator on the site has coordinates in it, and it owns everything that depends on the basis alone:
# the dual elements `project` takes overlaps with, the letter metric of the SVD backends, the lazily
# filled table of on-site letter products. `instantiate` is the inverse of `project` (projection.jl).

using TensorKit: AbstractTensorMap, ElementarySpace, numin, numout, insertrightunit, space,
    isomorphism, ComplexSpace, Trivial, dual, sectors, fuse
using LinearAlgebra: LinearAlgebra, Hermitian, eigvals, norm

"""
Default relative tolerance of `Basis` construction: elements parallel to (or, for the matrix-unit completion,
spanned by) earlier ones of their charge, and coefficients of a rank factorisation, are judged against it
relative to their norm / largest entry.
"""
const BASIS_RTOL = 1.0e-12

"""
    Basis(ops::AbstractVector; atol = 0, rtol = BASIS_RTOL)

A complete, ordered basis of on-site operators on one space `V`: the alphabet that [`project`](@ref)
expresses operators in. Each entry of `ops` is a charge-neutral operator `V ← V` (or a plain matrix,
without symmetry), used as an element, or `V ⊗ V ← V ⊗ V`, factorised into its charged single-site
pieces (for `S⁺ ⊗ S⁻` the pieces `S⁺` and `S⁻`).

Element 1 is the identity; single-site entries follow in the order given, then the factorised pieces,
then the matrix units that complete the basis to span every operator on `V`. An entry parallel to an
earlier element of its charge is dropped; the entries of one charge must be linearly independent
(checked) but need not be orthogonal; the matrix units already spanned are skipped.

`project` takes the overlap of an operator with the dual elements stored in the basis (the elements
`ẽₖ` of the same charge with `⟨ẽₖ, eₘ⟩ = δₖₘ`), which is why a basis is checked for completeness when it
is built.

Two bases are equal only if they are the same object: `==` is `===` and `hash` goes by identity. The
stock bases ([`spin_basis`](@ref), …) are memoised, so calling a builder twice returns the identical
object, and the sites of a lattice are told apart by identity.
"""
mutable struct Basis{I <: Sector, S <: ElementarySpace, T <: AbstractTensorMap}
    space::S
    elements::Vector{T}                        # basis order; `elements[1]` is the identity
    duals::Vector{T}                           # `duals[k]`: dual to `elements[k]` within its charge
    letters::Vector{Letter{I}}                 # the letter of each element
    bycharge::Dictionary{I, Vector{Int}}       # element positions per charge, in index order
    norms::Vector{Float64}                     # `norm(elements[k])`
    metric::Union{Nothing, Dictionary{I, NTuple{2, Matrix{ComplexF64}}}}   # see `lettermetric`
    products::Dict{Tuple{Letter{I}, Letter{I}}, SiteOperator{I}}           # filled lazily, letterproducts.jl
end

Base.:(==)(a::Basis, b::Basis) = a === b
Base.hash(B::Basis, h::UInt) = hash(objectid(B), h)

TensorKit.space(B::Basis) = B.space
TensorKit.sectortype(::Type{<:Basis{I}}) where {I} = I
Base.length(B::Basis) = length(B.elements)
Base.getindex(B::Basis, k::Integer) = B.elements[k]

function Base.show(io::IO, B::Basis)
    print(io, "Basis(", B.space, "; ", length(B), " elements: ")
    join(io, (string(c, " => ", length(ks)) for (c, ks) in pairs(B.bycharge)), ", ")
    return print(io, ")")
end

scalarop(c::Number, B::Basis) = scalarop(c, sectortype(B))

# `charge => number of letters`, what `checklattice` validates letters against
alphabet(B::Basis{I}, ::Type{I}) where {I} = Dict{I, Int}(c => length(ks) for (c, ks) in pairs(B.bycharge))

"""
    instantiate(op::Letter, B::Basis)
    instantiate(O::SiteOperator, B::Basis)

The element of `B` that the letter `op` names, `V ← V ⊗ Vect[I](c => 1)`, or the combination
`Σ coeff · element` of a symbolic on-site operator (whose letters must share one charge).
"""
function instantiate(op::Letter{I}, B::Basis{I}) where {I}
    return B.elements[_position(op, B)]
end

# The dual element of the letter: `⟨dualelement(l, B), instantiate(m, B)⟩ = δ_lm` for letters of one charge.
dualelement(op::Letter{I}, B::Basis{I}) where {I} = B.duals[_position(op, B)]

function _position(op::Letter{I}, B::Basis{I}) where {I}
    ks = get(B.bycharge, op.c, nothing)
    (ks === nothing || !(1 <= op.n <= length(ks))) && throw(
        ArgumentError("letter $op does not exist in the basis $B")
    )
    return ks[op.n]
end

_space(B::Basis) = space(B)
_lettertensor(letter::Letter, B::Basis) = instantiate(letter, B)

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
completed_elements(V, elems) = AbstractTensorMap[elems; [matrixunit(mu, V) for mu in matrixunits(V)]]

identityelement(V) = isomorphism(ComplexF64, V ← V ⊗ Vect[sectortype(V)](unit(sectortype(V)) => 1))

function Basis(ops::AbstractVector; atol::Real = 0, rtol::Real = BASIS_RTOL)
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
        append!(pieces, factorise_pair(h; atol, rtol))
    end
    # the given operators first, on their own: parallel ones are dropped, dependent ones throw
    kept, _, bycharge = _select_elements(V, [identityelement(V); singles; pieces]; rtol)
    _gramblocks(kept, bycharge, rtol)
    return basis_from_elements(V, completed_elements(V, kept); rtol, skipdependent = true)
end

# Charged elements (the first the identity) → complete `Basis`: group by charge, drop elements parallel to
# an earlier one, check independence and completeness, precompute the dual elements and the metric. With
# `skipdependent` an element in the span of the earlier ones of its charge is dropped instead of throwing.
function basis_from_elements(
        V::ElementarySpace, elems::AbstractVector; rtol::Real = BASIS_RTOL, skipdependent::Bool = false
    )
    I = sectortype(V)
    kept, letters, bycharge = _select_elements(V, elems; rtol, skipdependent)
    W = fuse(V ⊗ V')
    for c in sectors(W)
        n = length(get(bycharge, c, Int[]))
        n == dim(W, c) || throw(
            ArgumentError(
                "Basis: the elements of charge $c span $n of the $(dim(W, c)) dimensions of the " *
                    "operators of that charge on $V; a basis must be complete"
            )
        )
    end
    grams = _gramblocks(kept, bycharge, rtol)
    duals = similar(kept)
    for (c, ks) in pairs(bycharge)
        G = grams[c]
        Ginv = Matrix(inv(Hermitian(G)))
        # round-off from the inversion would otherwise become letters with ~1e-16 coefficients, each of
        # which costs the sweep a term
        for a in axes(Ginv, 1), b in axes(Ginv, 2)
            abs(Ginv[a, b]) <= rtol * sqrt(abs(Ginv[a, a] * Ginv[b, b])) && (Ginv[a, b] = 0)
        end
        for (j, k) in enumerate(ks)
            d = zero(kept[k])
            for (i, m) in enumerate(ks)
                iszero(Ginv[i, j]) || (d += Ginv[i, j] * kept[m])
            end
            duals[k] = d
        end
    end
    norms = Float64[norm(e) for e in kept]
    return Basis{I, typeof(V), eltype(kept)}(
        V, kept, duals, letters, bycharge, norms, lettermetric(grams), Dict{Tuple{Letter{I}, Letter{I}}, SiteOperator{I}}()
    )
end

function _select_elements(
        V::ElementarySpace, elems::AbstractVector; rtol::Real = BASIS_RTOL, skipdependent::Bool = false
    )
    I = sectortype(V)
    els = map(elems) do e
        x = zeros(ComplexF64, space(e))
        copy!(x, e)
        return x
    end
    first(els) ≈ identityelement(V) ||
        invariant_violation("the first element of a basis must be the identity")
    kept = similar(els, 0)
    letters = Letter{I}[]
    bycharge = Dictionary{I, Vector{Int}}()
    spans = Dict{I, Matrix{ComplexF64}}()    # per charge: orthonormal span of the kept elements
    for e in els
        space(e) == (V ← V ⊗ domain(e)[2]) && numin(e) == 2 ||
            invariant_violation("a basis element must be `V ← V ⊗ Vect[I](c => 1)`, got $(space(e))")
        c = only(sectors(domain(e)[2]))
        ks = get!(bycharge, c, Int[])
        any(ks) do k
            return abs2(inner(kept[k], e)) >= (1 - rtol) * real(inner(kept[k], kept[k])) * real(inner(e, e))
        end && continue
        if skipdependent
            a = ComplexF64[inner(matrixunit(mu, V), e) for mu in _chargeunits(V, c)]
            Q = get(spans, c, zeros(ComplexF64, length(a), 0))
            r = a - Q * (Q' * a)
            norm(r) > sqrt(rtol) * norm(a) || continue
            spans[c] = [Q r / norm(r)]
        end
        push!(kept, e)
        push!(ks, length(kept))
        push!(letters, Letter{I}(c, length(ks)))
    end
    filter!(!isempty, bycharge)    # a charge whose only candidates were dropped
    return kept, letters, bycharge
end

# Per charge, the Gram matrix `G[k, m] = ⟨eₖ, eₘ⟩` of the elements; throws if they are linearly dependent.
function _gramblocks(kept, bycharge::Dictionary{I}, rtol::Real) where {I}
    grams = Dictionary{I, Matrix{ComplexF64}}()
    for (c, ks) in pairs(bycharge)
        G = ComplexF64[inner(kept[k], kept[m]) for k in ks, m in ks]
        λ = eigvals(Hermitian(G))
        minimum(λ) > rtol * maximum(λ) || throw(
            ArgumentError(
                "Basis: the elements of charge $c are linearly dependent (Gram eigenvalues $λ); " *
                    "drop the redundant operator"
            )
        )
        insert!(grams, c, G)
    end
    return grams
end

# The matrix-unit letters of charge `c` on `V`.
_chargeunits(V, c) = filter(mu -> mu.c == c, matrixunits(V))

# Rank factorisation of a neutral two-site operator into charged single-site pieces. Per charge `c` of
# the left factor, `M[a, b]` is the coefficient on the matrix units `eₐ ⊗ e_b` and `M = C R` with `C` the
# pivot columns of `M` and `R` its reduced row echelon form; the pieces are the columns of `C` and rows
# of `R`, each scaled so its first entry above `rtol` is 1 (`S⁺ ⊗ S⁻ + S⁻ ⊗ S⁺` gives `S⁺`, `S⁻`, not a
# mixture). The scale is put back as the pair's coefficient by `project`.
function factorise_pair(h::AbstractTensorMap; atol::Real = 0, rtol::Real = BASIS_RTOL)
    V = codomain(h)[1]
    I = sectortype(V)
    bych = Dictionary{I, Vector{Term{I}}}()
    for t in project_units(h; atol).terms.terms
        push!(get!(bych, t.factors[1].letter.c, Term{I}[]), t)
    end
    blocks = map(sort!(collect(keys(bych)))) do c
        ts = bych[c]
        L = sort!(unique!([t.factors[1].letter for t in ts]))
        R = sort!(unique!([t.factors[2].letter for t in ts]))
        M = zeros(ComplexF64, length(L), length(R))
        for t in ts
            M[searchsortedfirst(L, t.factors[1].letter), searchsortedfirst(R, t.factors[2].letter)] += t.coeff
        end
        return (L, R, M)
    end
    θ = max(float(atol), rtol * maximum(((_, _, M),) -> maximum(abs, M), blocks; init = 0.0))
    pieces = AbstractTensorMap[]
    for (L, R, M) in blocks
        C, E = _echelon_factors(M, θ)
        for k in axes(C, 2)
            l, r = _leadingone(C[:, k], rtol), _leadingone(E[k, :], rtol)
            push!(pieces, sum(l[m] * matrixunit(L[m], V) for m in eachindex(L) if !iszero(l[m])))
            push!(pieces, sum(r[m] * matrixunit(R[m], V) for m in eachindex(R) if !iszero(r[m])))
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
