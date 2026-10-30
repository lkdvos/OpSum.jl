# Dense symmetric `TensorMap` → terms: `project` into a `Basis` by overlaps with its dual elements, and
# the basis-free expansion in matrix units (`project_units`) that only basis construction uses.
#
# No linear solve at projection time. The candidates `(ops, tree)` — letters at the sites coupled through a
# caterpillar tree — satisfy `inner(E_α, E_β) = δ_tree g Π_k G_k[ops, ops']` with `G_k` the Gram matrix of the
# elements of one charge and `g = dim(tot) / Π_k dim(ops[k].c)`. With the dual elements `ẽ` of a basis
# (`⟨ẽₖ, eₘ⟩ = δₖₘ`, stored in the `Basis`) the dual candidates `Ẽ_α` give `c_α = inner(Ẽ_α, h) / g`. For matrix
# units, orthonormal, the elements are their own duals.

using TensorKit: AbstractTensorMap, ElementarySpace, numin, numout, insertrightunit, oneunit
using LinearAlgebra: norm

# Relative coefficients below this are round-off from `inner`.
_default_rtol(h::AbstractTensorMap) = 100 * eps(real(float(scalartype(h))))

# `h` with the trailing total-charge leg the forward map produces, and that charge.
function _with_charge_leg(h::AbstractTensorMap, K::Int, ::Type{I}) where {I <: Sector}
    nin = numin(h)
    if nin == K
        return insertrightunit(h), unit(I)
    elseif nin == K + 1
        Vt = domain(h)[K + 1]
        secs = collect(sectors(Vt))
        length(secs) == 1 ||
            throw(ArgumentError("project: the total-charge leg must carry exactly one sector, got $Vt"))
        return h, only(secs)
    end
    throw(
        ArgumentError(
            "project: expected numin(h) == $K (charge-neutral) or $(K + 1) (with a trailing " *
                "total-charge leg), got $nin"
        )
    )
end

# The dual elements of a basis as the "site" of the dense oracle `dense`: letters stand for their dual elements.
struct _DualSite{B <: Basis}
    basis::B
end
_space(D::_DualSite) = space(D.basis)
_lettertensor(letter::Letter, D::_DualSite) = dualelement(letter, D.basis)

# The candidate letters of a site by charge, with the norm of each element: the matrix units of a bare space
# (orthonormal), the letters of a basis (the identity is among them, as element 1).
function _candidates(V::ElementarySpace)
    I = sectortype(V)
    groups = Dictionary{I, Vector{Tuple{Letter{I}, Float64}}}()
    for op in matrixunits(V)
        push!(get!(groups, op.c, Tuple{Letter{I}, Float64}[]), (op, 1.0))
    end
    return groups
end

function _candidates(D::_DualSite{<:Basis{I}}) where {I}
    B = D.basis
    return map(B.bycharge) do ks
        return Tuple{Letter{I}, Float64}[(B.letters[k], B.norms[k]) for k in ks]
    end
end

# `(ops, tree, coeff)` for every candidate whose norm contribution exceeds `θ`. `sites` are bare spaces
# (coordinates in matrix units) or `_DualSite`s (coordinates in the basis, `Ẽ` the dual candidate).
function _coefficients(hc::AbstractTensorMap, sites::AbstractVector, tot::I, θ::Real) where {I <: Sector}
    K = length(sites)
    bycharge = map(_candidates, sites)
    out = Tuple{Vector{Letter{I}}, FusionTree{I}, ComplexF64}[]
    for cs in Iterators.product(map(collect ∘ keys, bycharge)...)
        trees = caterpillar_trees(cs, tot)
        isempty(trees) && continue
        g = dim(tot) / prod(dim, cs)
        cands = ntuple(k -> bycharge[k][cs[k]], K)
        for pick in Iterators.product(cands...), tree in trees
            ops = Letter{I}[first(x) for x in pick]
            E = dense(ops, 1:K, tree, sites)
            c = ComplexF64(inner(E, hc) / g)
            # the norm contribution is `|c| √g Π‖eₖ‖`, so bare `abs(c)` would not control the dropped norm
            abs(c) * sqrt(g) * prod(last, pick) > θ || continue
            push!(out, (ops, tree, c))
        end
    end
    return out
end

# `h` with its trailing total-charge leg and that charge, after checking that it is a symmetric operator on
# the `K = length(Vs)` spaces `Vs`.
function _checked_charge_leg(h::AbstractTensorMap, Vs::AbstractVector{<:ElementarySpace})
    K = length(Vs)
    I = sectortype(h)
    requirefusion(canfuse, I, "project: GenericFusion sectors are deferred (fusion multiplicity > 1)")
    # `Vect[Trivial] === ComplexSpace`; this rejects e.g. `CartesianSpace`, which cannot carry a charge leg
    spacetype(h) === Vect[I] || throw(
        ArgumentError("project: `h` must live on $(Vect[I]) spaces, got $(spacetype(h))")
    )
    hc, tot = _with_charge_leg(h, K, I)
    # covers: domain ≠ codomain, a dual or higher-degeneracy charge leg
    Vprod = foldl(⊗, Vs)
    expected = Vprod ← (Vprod ⊗ Vect[I](tot => 1))
    space(hc) == expected ||
        throw(ArgumentError("project: unsupported operator space\n  got      $(space(hc))\n  expected $expected"))
    return hc, tot
end

function _terms(coeffs, ::Type{I}, K::Int) where {I}
    # a caterpillar tree is its running bond charges, i.e. the `Factor`s
    return Terms{I}(
        Term{I}[
            Term{I}(collect(1:K), Factor{I}[Factor{I}(ops[j], bondcharges(tree)[j]) for j in 1:K], c)
                for (ops, tree, c) in coeffs
        ]
    )
end

# The expansion of a symmetric `K`-site operator (optionally with a trailing total-charge leg) in
# matrix-unit letters, a `LocalOperator` on `K` slots: what basis construction factorises. Coefficients at
# or below `max(atol, rtol‖h‖)` are dropped. Every term is active on all slots (an identity factor is a
# trivial-charge matrix unit).
function project_units(h::AbstractTensorMap; atol::Real = 0, rtol::Real = _default_rtol(h))
    K = numout(h)
    K >= 1 ||
        throw(ArgumentError("project: `h` must act on at least one site, got numout(h) = $K"))
    Vs = [codomain(h)[k] for k in 1:K]
    hc, tot = _checked_charge_leg(h, Vs)
    θ = max(float(atol), float(rtol) * norm(hc))
    return LocalOperator{sectortype(h)}(_terms(_coefficients(hc, Vs, tot, θ), sectortype(h), K), K)
end

# Projection into a basis
# -----------------------

"""
    project(O, B::Basis; atol = 0, rtol = 100eps) -> SiteOperator     # O : V ← V, or a matrix
    project(h, B::Basis; atol = 0, rtol = 100eps) -> LocalOperator    # h : V^⊗K ← V^⊗K, K ≥ 2
    project(h, Bs::AbstractVector{<:Basis}; …)                        # slot j in Bs[j]

Express an operator in the basis `B`: a [`SiteOperator`](@ref) for one site, an unplaced
[`LocalOperator`](@ref) for `K` sites (placed with `h[i]` or `h[s₁, …, s_K]`). A slot a term does not
act on holds the identity, which placement drops. The coefficients are overlaps with the dual elements of
the basis, so `instantiate` of the result reproduces `h` up to round-off (a basis is complete); a trailing
total-charge leg is accepted. Coefficients below `max(atol, rtol‖h‖)` are dropped.
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
    length(Bs) == 1 || return LocalOperator{I}(out, length(Bs))
    ts = out.terms
    return SiteOperator{I}(Letter{I}[only(t.factors).letter for t in ts], ComplexF64[t.coeff for t in ts])
end

# `project` of a block on `length(Bs)` sites as a `LocalOperator`, also for one site.
project_local(h, Bs::AbstractVector{<:Basis{I}}) where {I} = LocalOperator{I}(_project_terms(h, Bs), length(Bs))

# The terms of `h` on the slots `1:K`, slot `j` in `Bs[j]`.
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
    hc, tot = _checked_charge_leg(h, [B.space for B in Bs])
    θ = max(float(atol), float(rtol) * norm(hc))
    return _terms(_coefficients(hc, [_DualSite(B) for B in Bs], tot, θ), I, K)
end
