# Projection: dense symmetric `TensorMap` → symbolic ITO terms
# ============================================================
# The inverse of `instantiate`. Writing a Hamiltonian by naming alphabet letters `(c, n)` is
# fragile — `n` indexes TensorKit's canonical block order, so permuting the sectors of `V` silently
# permutes every letter and you get a different operator with no error. `project` goes the other
# way: write the operator down as a `TensorMap` and let the expansion coefficients be computed.
#
# Why this needs no linear solve
# ------------------------------
# For sites `V_1…V_K` and total charge `tot`, the candidate basis is `α = (ops, tree)`: one letter
# per site from `instances(IrrepOperator, V_k)` and one caterpillar tree from `caterpillar_trees`,
# materialized by the forward map as `E_α = _instantiate_term(TermKey(1:K, ops, tree), Vs)`.
#
# Writing `E_α = Q_ops ∘ X_tree` as `_embed_caterpillar` builds it: `Q` is a permute of `⊗_k O_k`, so
# `‖Q‖ = Π_k ‖O_k‖ = 1`; `Q_ops† Q_ops'` contracts only site-local legs and factorizes per site into
# `Hom(Vect[c'=>1], Vect[c=>1])`, which vanishes unless `c == c'` and is one-dimensional otherwise —
# fixing that scalar from the K=1 case (where the alphabet is orthonormal) gives `δ/dim(c_k)` per
# site — and `Tr_q(X† X') = dim(tot)·δ_{tree,tree'}`. So the basis is *orthogonal* with
#
#     inner(E_α, E_β) = δ_ops · δ_tree · dim(tot) / Π_k dim(ops[k].c)   =:  δ_αβ · g(ops, tot)
#     c_α = inner(E_α, h) / g
#
# and the projection is a sequence of inner products, not a solve. `test_irrep_projection.jl` pins
# both the diagonal and the vanishing off-diagonal, including for fermionic sectors (where the
# per-site factorization above could in principle pick up braiding data, but does not).
#
# Counting candidates gives `Σ_{charges} (Π_k N_k(c_k))·#trees = mult_{⊗_k (V_k⊗V_k')}(tot)`, the
# dimension of the target homspace, with `N_k(c) = dim(fuse(V_k⊗V_k'), c)` exactly the enumeration
# range of `instances`. The basis is therefore complete: a symmetric `h` of the accepted shape is
# *always* exactly in the span, so the only real source of residual is `tol` truncation. The
# faithfulness check guards against that and against implementation bugs — not against
# non-representable input, of which there is none.

using TensorKit: AbstractTensorMap, ElementarySpace, numin, numout, insertrightunit, oneunit
using LinearAlgebra: norm

export project, matrixunit

# Coefficients below `100 * eps` (relative to `‖h‖`) are round-off from `inner`, not physics. Kept
# deliberately tight: dropping a genuinely small coupling would be silently absorbed by the
# faithfulness check, so the default must only ever discard noise.
_default_rtol(h::AbstractTensorMap) = 100 * eps(real(float(scalartype(h))))

# Bring `h` into the shape the forward map produces: physical legs plus a trailing total-charge leg.
# `insertrightunit` defaults to appending at the last domain slot, and the inserted space is
# `Vect[I](unit(I) => 1)` — bit-identical to what `_embed_field`/`_embed_caterpillar` build for
# `tot = unit(I)`. For a `TensorMap` it re-wraps the data unchanged, so it is free and exactly
# norm-preserving.
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
            E = _instantiate_term(TermKey{I, Int}(collect(1:K), ops, tree), Vs)
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

"""
    project(h::AbstractTensorMap, sites; atol = 0, rtol = 100eps) -> TermSum

Project a symmetric `K`-site operator onto the ITO term basis — the inverse of
[`instantiate`](@ref). `h` must have one of the two shapes `instantiate` produces, with
`K = length(sites) = numout(h)`:

    h :  V_1 ⊗ … ⊗ V_K  ←  V_1 ⊗ … ⊗ V_K                          # charge-neutral
    h :  V_1 ⊗ … ⊗ V_K  ←  V_1 ⊗ … ⊗ V_K ⊗ Vect[I](tot => 1)      # total charge `tot`

The physical spaces are read off `h`; `sites` only supplies the labels, and must be strictly
increasing (`couple`'s left-to-right convention, which the downstream sweep relies on). The
coefficients are exact inner products against an orthogonal, complete basis, so no fit is involved.

Coefficients whose norm contribution falls at or below `max(atol, rtol * norm(h))` are dropped;
the result is then re-materialized and compared against `h`, and an `ArgumentError` is thrown if
the residual exceeds that same tolerance. A projected `TermSum` therefore provably represents its
input. An operator that is zero (or entirely below tolerance) gives an empty `TermSum`.

Every returned term is active on **all** `K` sites: an on-site identity factor comes back as a
trivial-charge letter, not as a shorter term. So `project ∘ instantiate` is the identity only for
term sums whose terms all have full support on `sites`.

```jldoctest
julia> using TensorKit

julia> V = SU2Space(1//2 => 1);

julia> h = OpSum.instantiate(couple(spin(V)[1], spin(V)[2]; to = SU2Irrep(0)), [V, V]);

julia> H = project(h, [1, 2]);

julia> length(H.terms)
1
```

See also [`matrixunit`](@ref), [`couple`](@ref).
"""
function project(
        h::AbstractTensorMap, sites; atol::Real = 0, rtol::Real = _default_rtol(h)
    )
    sitev = collect(sites)
    K = numout(h)
    K >= 1 ||
        throw(ArgumentError("project: `h` must act on at least one site, got numout(h) = $K"))
    length(sitev) == K || throw(
        ArgumentError("project: got $(length(sitev)) site labels for an operator on $K sites")
    )
    all(i -> sitev[i] < sitev[i + 1], 1:(K - 1)) || throw(
        ArgumentError("project: `sites` must be strictly increasing (sorted and unique), got $sitev")
    )

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

    S = eltype(sitev)
    terms = Dictionary{TermKey{I, S}, ComplexF64}()
    local_terms = Dictionary{TermKey{I, Int}, ComplexF64}()
    local_sites = collect(1:K)
    for (ops, tree, c) in coeffs
        # candidates are distinct by construction, so no accumulation is needed
        insert!(terms, TermKey{I, S}(sitev, ops, tree), c)
        insert!(local_terms, TermKey{I, Int}(local_sites, ops, tree), c)
    end

    # Faithfulness: recompute the operator from the emitted keys alone. This is a genuinely
    # independent pass through the forward map, so it also catches a wrong `g` or a misassigned tree.
    resid = if isempty(local_terms)
        hnorm
    else
        norm(hc - instantiate(TermSum{I, Int, ComplexF64}(local_terms), Vs))
    end
    slack = 16 * eps(real(float(scalartype(hc)))) * sqrt(max(1, ncand)) * hnorm
    resid <= θ + slack || throw(
        ArgumentError(
            "project: the projected term sum is not faithful to `h` (residual $resid exceeds " *
                "$(θ + slack), ‖h‖ = $hnorm); lower `atol`/`rtol`"
        )
    )

    return TermSum{I, S, ComplexF64}(terms)
end

"""
    project(O::AbstractTensorMap, V::ElementarySpace; atol = 0, rtol = 100eps) -> LocalOp

Expand a single-site operator in the ITO alphabet of `V`, as a symbolic [`LocalOp`](@ref) that is
not yet placed on the lattice. `O` must be either `V ← V` or `V ← V ⊗ Vect[I](c => 1)`. The result
is a `Sum` variant with one term per surviving letter, even when that is a single letter.

Because a `LocalOp` supports ordinary arithmetic, composite on-site operators read the way you
would write them on paper, and `A[i]` then places the whole expansion:

```julia
Sz = (matrixunit(V, up, up) - matrixunit(V, dn, dn)) / 2
H  = sum(couple(Sz[i], Sz[i + 1]; to = unit(I)) for i in 1:(N - 1))
```

Note that the identity is not an alphabet letter: `project(id(V), V)` returns a sum of
trivial-charge letters rather than `scalarop(1, V)`. An operator that is zero (or entirely below
tolerance) gives the zero scalar `LocalOp`, which does not retain the charge of the input.
"""
function project(
        O::AbstractTensorMap, V::ElementarySpace; atol::Real = 0, rtol::Real = _default_rtol(O)
    )
    numout(O) == 1 || throw(
        ArgumentError("project: a single-site operator must have one output leg, got $(numout(O))")
    )
    codomain(O)[1] == V ||
        throw(ArgumentError("project: `O` acts on $(codomain(O)[1]), not on the given space $V"))

    A = IrrepOperator{sectortype(V)}
    ts = project(O, (1,); atol, rtol)
    isempty(ts.terms) && return zero(LocalOp{ComplexF64, A})
    return sum(v * LocalOp{ComplexF64, A}(only(k.ops)) for (k, v) in pairs(ts.terms))
end

"""
    matrixunit(V::ElementarySpace, out::I, in::I) -> LocalOp

The matrix unit ``|out⟩⟨in|`` on a space whose sectors are one-dimensional and appear with
multiplicity one (abelian or fermionic: `Vect[U₁]`, `Vect[FermionNumber]`, and products thereof).
The operator charge follows from the fusion `out ⊗ in' → c`, and the alphabet letters are *derived*
by [`project`](@ref) rather than named by index.

```julia
V = Vect[FermionNumber](0 => 1, 1 => 1)
vac, occ = FermionNumber(0), FermionNumber(1)
c  = matrixunit(V, vac, occ)   # annihilation
cd = matrixunit(V, occ, vac)   # creation
n  = matrixunit(V, occ, occ)   # number
```
"""
function matrixunit(V::ElementarySpace, out::I, in::I) where {I <: Sector}
    dim(out) == 1 && dim(in) == 1 ||
        throw(ArgumentError("matrixunit requires one-dimensional sectors, got $out and $in"))
    dim(V, out) == 1 && dim(V, in) == 1 ||
        throw(ArgumentError("matrixunit requires multiplicity-one sectors in $V, got $out and $in"))
    c = only(⊗(out, dual(in)))
    t = zeros(ComplexF64, V ← V ⊗ Vect[I](c => 1))
    block(t, out) .= 1
    return project(t, V)
end
