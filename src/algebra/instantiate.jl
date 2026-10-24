# The dense-tensor correctness oracle behind `instantiate(::OperatorSum)`, the inverse of `project`
# (`OpSum.dense`, internal).

using TensorKit
using TensorKit: ElementarySpace, FusionTree, unit, dim, id, Vect, domain, permute, insertrightunit

# The dense oracle takes each site as a bare space (matrix-unit letters, as in `project`) or as a `Basis`.
_space(V::ElementarySpace) = V
_lettertensor(letter::Letter, V::ElementarySpace) = matrixunit(letter, V)

# The sum of the terms of `h` on `sites` (spaces or bases).
function dense(h::Terms, sites)
    return sum(t -> t.coeff * dense(t, sites), check_resolved(h).terms)
end

# The term with letters `ops` at `positions` and fusion `tree`, on `sites` (spaces or bases).
function dense(ops, positions, tree, sites)
    # a K = 0 term needs the trailing charge leg too, or it cannot be summed with the others
    isempty(ops) && return insertrightunit(foldl(⊗, (id(_space(V)) for V in sites)))
    return _embed_caterpillar(ops, positions, tree, sites)
end

dense(t::Term, sites) = dense(letters(t), t.sites, tree(t), sites)

# charged letter on site `p`, identities elsewhere, charge leg last in the domain (one factor of a written product)
function embed_letter(op::Letter, p, sites)
    N = length(sites)
    loc = _lettertensor(op, sites[p])                # V_p ← V_p ⊗ V_c
    full = foldl(⊗, (j == p ? loc : id(_space(sites[j])) for j in 1:N))
    cod = ntuple(identity, N)
    charge_global = N + (p + 1)
    dom_wo_charge = (ntuple(m -> N + m, p)..., ntuple(m -> N + p + 1 + m, N - p)...)
    dom = (dom_wo_charge..., charge_global)
    return permute(full, (cod, dom))
end

# Caterpillar K-site block; the coupler `X` selects the channel `tree`.
function _embed_caterpillar(ops, positions, tree, sites)
    K = length(ops)
    I = typeof(tree.coupled)
    Os = [_lettertensor(ops[k], sites[positions[k]]) for k in 1:K]   # V ← V ⊗ Vc_k
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

    # embed the block (o over positions, i over positions, total charge) in the full lattice in site
    # order, total-charge leg last
    N = length(sites)
    idle = [k for k in 1:N if !(k in positions)]
    full = isempty(idle) ? Wblock : Wblock ⊗ foldl(⊗, (id(_space(sites[k])) for k in idle))

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
