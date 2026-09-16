# The dense-tensor correctness oracle: instantiate(H::TermSum) and its embed helpers, the inverse of `project`.

using TensorKit
using TensorKit: ElementarySpace, FusionTree, unit, dim, id, Vect, domain, permute, insertrightunit
using .IrrepTensorOperators: IrrepOperator

"""
    instantiate(H::TermSum)
    instantiate(ts::Terms, sites::AbstractVector{<:ElementarySpace})

Materialize the operator into a TensorKit `TensorMap` over its lattice (the dense oracle), summing
each term. Supports identity (K=0), single-site field (K=1), and left-nested (caterpillar) coupling
of any K ≥ 2 sites.
"""
function instantiate(H::TermSum)
    isempty(H) && throw(ArgumentError("cannot instantiate an empty TermSum"))
    sites = H.lattice
    length(sites) == 0 && throw(ArgumentError("cannot instantiate over an empty lattice"))
    return sum(t -> t.coeff * _instantiate_term(t, sites), H.terms)
end
instantiate(ts::Terms, sites::AbstractVector{<:ElementarySpace}) = instantiate(opsum(sites, ts))

# Shared forward map for both `_instantiate_term` (from a `Term`) and `_instantiate_basis` (from raw
# letters/positions/tree, which `project` takes inner products against).
function _instantiate_generic(ops, positions, tree, sites)
    K = length(ops)
    # The trailing total-charge leg is `Vect[I](unit(I) => 1)` for every K ≥ 1 term, so a K = 0
    # identity term needs one too — otherwise an operator mixing the two cannot be summed at all.
    K == 0 && return insertrightunit(foldl(⊗, (id(V) for V in sites)))
    K == 1 && return _embed_field(only(ops), only(positions), sites)
    return _embed_caterpillar(ops, positions, tree, sites)
end

_instantiate_term(t::Term, sites) = _instantiate_generic(ops(t), t.sites, tree(t), sites)

# The candidate basis element for a `(letters, tree)` combination on the local lattice `1:K`.
_instantiate_basis(ops, tree, sites) = _instantiate_generic(ops, 1:length(ops), tree, sites)

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

# Caterpillar K-site block; the coupler `X` selects the specific channel `tree`. `permute` +
# composition rather than `@tensor`, so it generalises to any K.
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
