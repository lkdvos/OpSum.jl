# Finite compression behind `mpo`, `mpo_terms` (its inverse on the reduced level) and the dense assembly
# `mpo_tensors`.

using SparseArrays: SparseMatrixCSC
using TensorKit: FusionTree, Vect, fusiontrees, permute, unit, @tensor, ncon, removeunit

# Bond matrices, bond sectors and per-bond identity-channel indices (empty with decaying channels, where
# they are unknown). Without channels `sweep` forces the finish channel into the cover.
function _compress(H::OperatorSum{I, <:FiniteChain}, alg::BondStrategy) where {I}
    N = length(H.lattice)
    if isempty(H.channels)
        # a truncation has to see the basis' own inner product (metric.jl)
        truncates(alg) && return compress_in_metric(H.terms, H.lattice, alg)
        return sweep(TermTable(H.terms), N, alg)
    end
    return _compress_channels(H, alg)
end

function _compress_channels(H::OperatorSum{I, <:FiniteChain}, ::VertexCover) where {I}
    N = length(H.lattice)
    (; Ws, bondsectors) = sweep(
        TermTable(H.terms), N, VertexCover(); jordan = false, channels = lower_channels(H.channels, 1)
    )
    return (; Ws, bondsectors, start = Int[], finish = Int[])
end
_compress_channels(::OperatorSum, ::Union{IndependentSVD, SequentialSVD}) = throw(
    ArgumentError(
        "SVD strategies cannot compress an exponentially decaying interaction: they need the " *
            "whole bond coefficient matrix up front. Use VertexCover() (the default)."
    )
)

"""
    mpo_terms(W::SymbolicMPO) -> Terms

Reconstruct the operator `W` generates, as a latticeless [`Terms`](@ref): enumerate every path through the
bond matrices and read off the letters with their outgoing bond charges. `mpo_terms(mpo(H)) ≈ H.terms` is
[`islossless`](@ref). Finite chains only.
"""
mpo_terms(W::SymbolicMPO{I, <:FiniteChain}) where {I} = reduced_terms(W.Ws, W.bondsectors)

# The paths of the bond matrices `Ws` from index `leftidx` of the first bond to index `rightidx` of the last
# (default `1` and "any"); for an infinite MPO tiled over a window they are the two identity channels.
function reduced_terms(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}},
        bondsectors::Vector{Vector{I}};
        leftidx::Int = 1, rightidx::Union{Nothing, Int} = nothing
    ) where {I}
    N = length(Ws)
    N == 0 && return Terms{I}()
    cols = Term{I}[]
    function walk(i, l0, coeff, sites, keys)
        if i > N
            (rightidx === nothing || l0 == rightidx) &&
                push!(cols, Term{I}(copy(sites), copy(keys), coeff))
            return
        end
        for (idx, localop) in storedpairs(Ws[i])
            l, r = Tuple(idx)
            l == l0 || continue
            for (letter, c) in pairs(localop)
                pass = ispassthrough(letter)
                pass || (push!(sites, i); push!(keys, Factor{I}(letter, bondsectors[i][r])))
                walk(i + 1, r, coeff * c, sites, keys)
                pass || (pop!(sites); pop!(keys))
            end
        end
        return
    end
    walk(1, leftidx, ComplexF64(1), Int[], Factor{I}[])
    return Terms{I}(cols)
end

# Site tensors `W_i : B_{i-1} ⊗ V_i ← V_i ⊗ B_i` (MPSKit leg convention).

# bond space with the given per-index charges
function _bond_space(sec::Vector{I}) where {I}
    counts = Dict{I, Int}()
    for c in sec
        counts[c] = get(counts, c, 0) + 1
    end
    return Vect[I](counts)
end

# degeneracy index (within its charge sector) of each bond index
function _deg_indices(sec::Vector{I}) where {I}
    counts = Dict{I, Int}()
    return [counts[c] = get(counts, c, 0) + 1 for c in sec]
end

# One letter's block for a single `(bL, bR)` bond-charge pair (no bond degeneracy), for `jordan_mpo_tensors`.
function _letter_block(letter::Letter{I}, site::Basis{I}, bL::I, bR::I) where {I}
    c = letter.c
    V = space(site)
    f1 = only(fusiontrees((bL, c), bR, (false, false)))
    f2 = only(fusiontrees((bR,), bR, (false,)))
    κ = zeros(ComplexF64, Vect[I](bL => 1) ⊗ Vect[I](c => 1) ← Vect[I](bR => 1))
    κ[f1, f2][1, 1, 1] = 1
    blk = zeros(ComplexF64, Vect[I](bL => 1) ⊗ V ← V ⊗ Vect[I](bR => 1))
    O = instantiate(letter, site)
    @tensor blk[bl o; ii br] += O[o; ii cc] * κ[bl cc; br]
    return blk
end

"""
    mpo_tensors(W::SymbolicMPO) -> Vector{<:AbstractTensorMap}

Dense symmetric site tensors `W_i : B_{i-1} ⊗ V_i ← V_i ⊗ B_i` (MPSKit convention) of an [`mpo`](@ref), each
entry read in its site's basis; the boundary bonds of a finite chain are one-dimensional. On an
[`InfiniteChain`](@ref) they are the `L` tensors of the unit cell, which tile.
"""
mpo_tensors(W::SymbolicMPO) = reduced_tensors(W.Ws, W.bondsectors, W.lattice)

# The tensors of bond matrices `Ws` with the bond charges `bondsectors` of the bonds right of each site, on
# `lat` (or a finite chain of the given sites).
function reduced_tensors(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}},
        bondsectors::Vector{Vector{I}}, lat::FiniteChain
    ) where {I}
    return _mpo_tensors(Ws, vcat([I[unit(I)]], bondsectors), lat)
end
# the unit cell: the bond left of site 1 is bond `L`, so the tensors tile
function reduced_tensors(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}},
        bondsectors::Vector{Vector{I}}, lat::InfiniteChain
    ) where {I}
    L = length(Ws)
    return _mpo_tensors(Ws, Vector{I}[bondsectors[mod1(j - 1, L)] for j in 1:(L + 1)], lat)
end
function reduced_tensors(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}},
        bondsectors::Vector{Vector{I}}, sites::AbstractVector{<:Basis}
    ) where {I}
    return reduced_tensors(Ws, bondsectors, FiniteChain(sites))
end

# `bsecs[i]` is the bond to the *left* of site i (length N+1); finite and infinite differ in `bsecs[1]`.
function _mpo_tensors(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}},
        bsecs::Vector{Vector{I}}, sites
    ) where {I}
    N = length(Ws)
    bspaces = map(_bond_space, bsecs)
    bdegs = map(_deg_indices, bsecs)
    return map(1:N) do i
        return _site_tensor(sites[i], Ws[i], bsecs[i], bsecs[i + 1], bspaces[i], bspaces[i + 1], bdegs[i], bdegs[i + 1])
    end
end

# The tensor of one site; the arguments are concrete here, so the caches below are typed.
function _site_tensor(
        site::Basis{I, S, T}, M::SparseMatrixCSC{SiteOperator{I}}, secL::Vector{I}, secR::Vector{I},
        Bleft, Bright, degL::Vector{Int}, degR::Vector{Int}
    ) where {I, S, T}
    V = space(site)
    W = zeros(ComplexF64, Bleft ⊗ V ← V ⊗ Bright)

    # per *letter*: κ collects every coefficient carrying it, so `O(d²)` contractions, not `O(D_L D_R)`
    ops = Dictionary{Letter{I}, T}()
    couplers = Dictionary{Letter{I}, typeof(zeros(ComplexF64, Bleft ⊗ Vect[I](unit(I) => 1) ← Bright))}()
    f1s = Dictionary{Tuple{I, I, I}, FusionTree{I, 2, 0, 1}}()
    f2s = Dictionary{I, FusionTree{I, 1, 0, 0}}()

    for (idx, localop) in storedpairs(M)
        l, r = Tuple(idx)
        bL, dL = secL[l], degL[l]
        bR, dR = secR[r], degR[r]
        for (letter, coeff) in pairs(localop)
            c = letter.c
            get!(() -> instantiate(letter, site), ops, letter)
            # running bond FIRST, matching the caterpillar (charge-first flips the sign at 1⊗1→1)
            κ = get!(couplers, letter) do
                return zeros(ComplexF64, Bleft ⊗ Vect[I](c => 1) ← Bright)
            end
            f1 = get!(() -> only(fusiontrees((bL, c), bR, (false, false))), f1s, (bL, c, bR))
            f2 = get!(() -> only(fusiontrees((bR,), bR, (false,))), f2s, bR)
            κ[f1, f2][dL, 1, dR] += coeff
        end
    end

    for (letter, κ) in pairs(couplers)
        O = ops[letter]
        @tensor W[bl o; ii br] += O[o; ii cc] * κ[bl cc; br]
    end
    return W
end

"""
    islossless(H::OperatorSum[, alg]) -> Bool

Whether the compressed MPO reproduces `H` exactly: `mpo_terms(mpo(H)) ≈ chain_terms(H)`. Purely symbolic,
finite chains only; after a truncating SVD [`BondStrategy`](@ref) `false` is expected.
"""
function islossless(H::OperatorSum{I, <:FiniteChain}, args...) where {I}
    return mpo_terms(mpo(H, args...)) ≈ chain_terms(H)
end

islossless(::OperatorSum{I, <:InfiniteChain}, args...) where {I} = throw(
    ArgumentError("islossless is not supported on an infinite chain")
)

"""
    mpo_tensormap(Ts::AbstractVector) -> AbstractTensorMap

Contract MPO site tensors ([`mpo_tensors`](@ref), or [`jordan_mpo_tensors`](@ref) after
`map(TensorMap, ·)`) into the `N`-site operator in [`instantiate`](@ref)'s leg convention, so that
`mpo_tensormap(mpo_tensors(mpo(H))) ≈ instantiate(H)`. Exponential in `N`.
"""
function mpo_tensormap(Ts::AbstractVector)
    N = length(Ts)
    N >= 1 || throw(ArgumentError("mpo_tensormap: need at least one site tensor"))
    net = [[i == 1 ? -(2N + 1) : i - 1, -i, -(N + i), i == N ? -(2N + 2) : i] for i in 1:N]
    O = removeunit(ncon(Ts, net), 2N + 1)    # legs o₁…o_N, i₁…i_N, bL, bR; drop the left boundary
    return permute(O, (ntuple(identity, N), (ntuple(i -> N + i, N)..., 2N + 1)))
end
