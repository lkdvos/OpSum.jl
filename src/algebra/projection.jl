# Dense symmetric `TensorMap` → matrix-unit terms, the inverse of `instantiate`. No linear solve:
# the candidates `(ops, tree)` are orthogonal and complete with `inner(E_α, E_β) = δ_αβ g`,
# `g = dim(tot) / Π_k dim(ops[k].c)`, so `c_α = inner(E_α, h) / g`.

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

# `(ops, tree, coeff)` for every candidate whose norm contribution exceeds `θ`, and the candidate count.
function _coefficients(
        hc::AbstractTensorMap, Vs::AbstractVector{<:ElementarySpace}, tot::I, θ::Real
    ) where {I <: Sector}
    K = length(Vs)
    bycharge = map(Vs) do V
        groups = Dictionary{I, Vector{Letter{I}}}()
        for op in matrixunits(V)
            haskey(groups, op.c) ? push!(groups[op.c], op) : insert!(groups, op.c, [op])
        end
        return groups
    end

    out = Tuple{Vector{Letter{I}}, FusionTree{I}, ComplexF64}[]
    ncand = 0
    for cs in Iterators.product(map(collect ∘ keys, bycharge)...)
        trees = caterpillar_trees(cs, tot)
        isempty(trees) && continue
        g = dim(tot) / prod(dim, cs)
        letters = ntuple(k -> bycharge[k][cs[k]], K)
        for opstup in Iterators.product(letters...), tree in trees
            ncand += 1
            ops = collect(Letter{I}, opstup)
            E = _instantiate_term(ops, 1:K, tree, Vs)
            c = ComplexF64(inner(E, hc) / g)
            # the norm contribution is `|c| √g`, so bare `abs(c)` would not control the dropped norm
            abs(c) * sqrt(g) > θ || continue
            push!(out, (ops, tree, c))
        end
    end
    return out, ncand
end

# The expansion of a symmetric `K`-site operator (optionally with a trailing total-charge leg) in
# matrix-unit letters, a `LocalOperator` on `K` slots: the coordinates every `Basis` is projected
# through. Coefficients at or below `max(atol, rtol‖h‖)` are dropped; the result is checked against `h`.
# Every term is active on all slots (an identity factor is a trivial-charge matrix unit).
function _project_units(h::AbstractTensorMap; atol::Real = 0, rtol::Real = _default_rtol(h))
    K = numout(h)
    K >= 1 ||
        throw(ArgumentError("project: `h` must act on at least one site, got numout(h) = $K"))

    I = sectortype(h)
    FusionStyle(I) isa GenericFusion && throw(
        ArgumentError("project: GenericFusion sectors ($I) are deferred (fusion multiplicity > 1)")
    )
    # `Vect[Trivial] === ComplexSpace`; this rejects e.g. `CartesianSpace`, which cannot carry a charge leg
    spacetype(h) === Vect[I] || throw(
        ArgumentError("project: `h` must live on $(Vect[I]) spaces, got $(spacetype(h))")
    )

    Vs = [codomain(h)[k] for k in 1:K]
    hc, tot = _with_charge_leg(h, K, I)

    # covers: domain ≠ codomain, a dual or higher-degeneracy charge leg
    Vprod = foldl(⊗, Vs)
    expected = Vprod ← (Vprod ⊗ Vect[I](tot => 1))
    space(hc) == expected ||
        throw(ArgumentError("project: unsupported operator space\n  got      $(space(hc))\n  expected $expected"))

    hnorm = norm(hc)
    θ = max(float(atol), float(rtol) * hnorm)
    coeffs, ncand = _coefficients(hc, Vs, tot, θ)

    # a caterpillar tree is its running bond charges, i.e. the `Factor`s
    out = Terms{I}(
        Term{I}[
            Term{I}(collect(1:K), Factor{I}[Factor{I}(ops[j], bondcharges(tree)[j]) for j in 1:K], c)
                for (ops, tree, c) in coeffs
        ]
    )

    # faithfulness: recompute the operator from the emitted terms (also catches a wrong `g` or tree)
    resid = isempty(coeffs) ? hnorm : norm(hc - _instantiate_terms(out, Vs))
    slack = 16 * eps(real(float(scalartype(hc)))) * sqrt(max(1, ncand)) * hnorm
    resid <= θ + slack || throw(
        ArgumentError(
            "project: the projected term sum is not faithful to `h` (residual $resid exceeds " *
                "$(θ + slack), ‖h‖ = $hnorm); lower `atol`/`rtol`"
        )
    )

    return LocalOperator{I}(out, K)
end
