# Projection: dense symmetric `TensorMap` → symbolic ITO terms, the inverse of `instantiate`. Naming
# letters `(c, n)` directly is fragile since `n` follows TensorKit's block order, so operators are
# built as tensors and projected back instead.
#
# No linear solve needed: the candidate basis `α = (ops, tree)` is orthogonal with a closed-form
# diagonal, so `c_α = inner(E_α, h) / g` where
#
#     inner(E_α, E_β) = δ_ops · δ_tree · dim(tot) / Π_k dim(ops[k].c)  =:  δ_αβ · g(ops, tot)
#
# and complete (candidate count = target homspace dimension), so a symmetric `h` of the accepted shape
# is always exactly in the span; the only residual is `tol` truncation.

using TensorKit: AbstractTensorMap, ElementarySpace, numin, numout, insertrightunit, oneunit
using LinearAlgebra: norm

# Coefficients below `100 * eps` (relative to `‖h‖`) are round-off from `inner`, not physics. Kept
# deliberately tight: dropping a genuinely small coupling would be silently absorbed by the
# faithfulness check, so the default must only ever discard noise.
_default_rtol(h::AbstractTensorMap) = 100 * eps(real(float(scalartype(h))))

# Give `h` the trailing total-charge leg the forward map produces. `insertrightunit` re-wraps the data
# unchanged, so it is free and exactly norm-preserving.
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

# The projection proper: `(ops, tree, coeff)` for every candidate whose norm contribution exceeds
# `θ`, plus the total number of candidates considered (for the round-off slack).
function _ito_coefficients(
        hc::AbstractTensorMap, Vs::AbstractVector{<:ElementarySpace}, tot::I, θ::Real
    ) where {I <: Sector}
    K = length(Vs)
    # letters grouped by charge, once per site
    bycharge = map(Vs) do V
        groups = Dictionary{I, Vector{IrrepOperator{I}}}()
        for op in instances(IrrepOperator, V)
            haskey(groups, op.c) ? push!(groups[op.c], op) : insert!(groups, op.c, [op])
        end
        return groups
    end

    out = Tuple{Vector{IrrepOperator{I}}, FusionTree{I}, ComplexF64}[]
    ncand = 0
    for cs in Iterators.product(map(collect ∘ keys, bycharge)...)
        trees = caterpillar_trees(cs, tot)
        isempty(trees) && continue
        # g = inner(E, E), analytic (an exact ratio of quantum dimensions — no extra contraction)
        g = dim(tot) / prod(dim, cs)
        letters = ntuple(k -> bycharge[k][cs[k]], K)
        for opstup in Iterators.product(letters...), tree in trees
            ncand += 1
            ops = collect(IrrepOperator{I}, opstup)
            E = _instantiate_basis(ops, tree, Vs)
            c = ComplexF64(inner(E, hc) / g)
            # Threshold the component's *norm contribution*: by orthogonality
            # `‖dropped‖² = Σ|c_α|² g_α` exactly, and `g` varies by `dim(tot)/Π dim(c_k)`, so bare
            # `abs(c)` is not norm-controlled.
            abs(c) * sqrt(g) > θ || continue
            push!(out, (ops, tree, c))
        end
    end
    return out, ncand
end

# The exact expansion of a symmetric `K`-site operator in the matrix-unit letters of its spaces, as an
# unplaced `LocalOperator` on `K` slots: the coordinates every `Basis` is defined and projected
# through (basis.jl). `h` is `V₁ ⊗ … ⊗ V_K ← V₁ ⊗ … ⊗ V_K`, optionally with a trailing total-charge
# leg. The coefficients are exact inner products against an orthogonal, complete basis; those at or
# below `max(atol, rtol‖h‖)` are dropped, and the result is re-materialised and checked against `h`.
# Every term is active on all `K` slots (an identity factor is a trivial-charge matrix unit). Not
# public: the matrix-unit alphabet is internal since operator-bases.md §7, step 7.
function _project_units(h::AbstractTensorMap; atol::Real = 0, rtol::Real = _default_rtol(h))
    K = numout(h)
    K >= 1 ||
        throw(ArgumentError("project: `h` must act on at least one site, got numout(h) = $K"))

    I = sectortype(h)
    FusionStyle(I) isa GenericFusion && throw(
        ArgumentError("project: GenericFusion sectors ($I) are deferred (fusion multiplicity > 1)")
    )
    # `Vect[Trivial] === ComplexSpace`, so this accepts unsymmetric complex spaces but rejects e.g.
    # `CartesianSpace`, which shares their sector type yet cannot carry a charge leg.
    spacetype(h) === Vect[I] || throw(
        ArgumentError("project: `h` must live on $(Vect[I]) spaces, got $(spacetype(h))")
    )

    Vs = [codomain(h)[k] for k in 1:K]
    hc, tot = _with_charge_leg(h, K, I)

    # One equality covers the remaining shape errors: domain ≠ codomain, a dual or higher-degeneracy
    # charge leg, and a wrong space type.
    Vprod = foldl(⊗, Vs)
    expected = Vprod ← (Vprod ⊗ Vect[I](tot => 1))
    space(hc) == expected ||
        throw(ArgumentError("project: unsupported operator space\n  got      $(space(hc))\n  expected $expected"))

    hnorm = norm(hc)
    θ = max(float(atol), float(rtol) * hnorm)
    coeffs, ncand = _ito_coefficients(hc, Vs, tot, θ)

    # A caterpillar tree *is* its running bond charges plus vertex labels, i.e. the `ITOKey`s.
    # Candidates are distinct by construction, so nothing accumulates.
    slots = collect(1:K)
    terms = Term{I}[]
    sizehint!(terms, length(coeffs))
    for (ops, tree, c) in coeffs
        bonds, verts = bondcharges(tree), vertexlabels(tree)
        keys = ITOKey{I}[ITOKey{I}(ops[j], bonds[j], verts[j]) for j in 1:K]
        push!(terms, Term{I}(slots, keys, c))
    end
    out = Terms{I}(terms)

    # Faithfulness: recompute the operator from the emitted terms alone. This is a genuinely
    # independent pass through the forward map, so it also catches a wrong `g` or a misassigned tree.
    resid = isempty(terms) ? hnorm : norm(hc - _instantiate_terms(out, Vs))
    slack = 16 * eps(real(float(scalartype(hc)))) * sqrt(max(1, ncand)) * hnorm
    resid <= θ + slack || throw(
        ArgumentError(
            "project: the projected term sum is not faithful to `h` (residual $resid exceeds " *
                "$(θ + slack), ‖h‖ = $hnorm); lower `atol`/`rtol`"
        )
    )

    return LocalOperator{I}(out, K)
end
