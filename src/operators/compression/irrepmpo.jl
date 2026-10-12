# Reduced MPO for the ITO automaton: `mpo` (symbolicmpo.jl) compresses an `OperatorSum` into reduced bond
# matrices plus per-bond charge sectors by a per-bond-sector sweep (irrepgraph.jl, strategy chosen by
# `algorithms.jl`) over the flat `ITOTermTable`. `mpo_terms` inverts it at the reduced level and
# `irrep_mpo_tensors` assembles the symmetric `TensorMap`s.

using SparseArrays: SparseMatrixCSC
using TensorKit: Vect, fusiontrees, permute, unit, @tensor, ncon, removeunit
using .IrrepTensorOperators: IrrepOperator

# The finite compression behind `mpo`: bond matrices, bond sectors and the two identity-channel indices
# per bond (empty when unknown, i.e. with decaying channels). Without channels it goes through
# `_irrep_channels`, which forces the finish channel into the cover so that the result has a Jordan form.
function _compress(
        H::OperatorSum{I, <:FiniteChain}, alg::Union{BipartiteAlgorithm, SVDBondAlgorithm}
    ) where {I}
    N = length(H.lattice)
    if isempty(H.channels)
        # a truncation has to see the basis' own inner product (metric.jl)
        if alg isa SVDBondAlgorithm && alg.strategy.trunc !== nothing
            out = _compress_in_metric(H.terms, H.lattice, bondstrategy(alg))
            out === nothing || return out
        end
        tt = ITOTermTable(H.terms, N)
        return _irrep_channels(tt, nvertices(tt), bondstrategy(alg))
    end
    alg isa SVDBondAlgorithm && throw(
        ArgumentError(
            "SVDBondAlgorithm cannot compress an exponentially decaying interaction: it needs the " *
                "whole bond coefficient matrix up front, and a geometric channel is a bond index " *
                "with a diagonal rather than a column of it. Use BipartiteAlgorithm() (the default)."
        )
    )
    Ws, secs = _irrep_sweep(
        ITOTermTable(H.terms, N), N, VertexCover(); channels = _lower_channels(H.channels, 1)
    )
    return Ws, secs, Int[], Int[]
end

"""
    mpo_terms(Ws, bondsectors; leftidx = 1, rightidx = nothing) -> Terms

Reconstruct the operator a reduced MPO generates (inverse of [`mpo`](@ref) at the reduced level): enumerate
every path through the bond matrices, multiply the coefficients along it, and read the active ITO letters
with their outgoing bond charges (the caterpillar running bonds). The result is a latticeless
[`Terms`](@ref); for lossless compression `mpo_terms(mpo(H)) ≈ H.terms`, which [`islossless`](@ref)
packages up.

`leftidx` and `rightidx` are the boundary indices: by default `1` and "any final index" (a finite chain's
last bond is one-dimensional); for an infinite MPO tiled over a window they are the two identity channels.
"""
function mpo_terms(
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
                if ispassthrough(letter)
                    walk(i + 1, r, coeff * c, sites, keys)
                else
                    push!(sites, i)
                    push!(keys, ITOKey{I}(letter, bondsectors[i][r], 1))
                    walk(i + 1, r, coeff * c, sites, keys)
                    pop!(sites)
                    pop!(keys)
                end
            end
        end
        return
    end
    walk(1, leftidx, ComplexF64(1), Int[], ITOKey{I}[])
    return Terms{I}(cols)
end

# Site tensors `W_i : B_{i-1} ⊗ V_i ← V_i ⊗ B_i` (MPSKit leg convention).

# GradedSpace on a bond with the given per-index charges (multiplicity = count per sector).
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

# Function barriers: keep `@tensor` and the per-entry loop specialised on concrete types although the
# memo tables below are `Any`-typed.
function _add_bond_entry!(W, O, κ)
    @tensor W[bl o; ii br] += O[o; ii cc] * κ[bl cc; br]
    return W
end
function _add_coupler_coeff!(κ, f1, f2, dL::Int, dR::Int, coeff)
    κ[f1, f2][dL, 1, dR] += coeff
    return κ
end

# One letter's site-tensor block for a single `(bL, bR)` bond-charge pair with coefficient `1`: the
# on-site tensor coupled `(bL, c) → bR` (running bond first, matching the caterpillar), for
# `jordan_mpo_tensors`, which has no bond degeneracy to batch over.
function _letter_block(letter::IrrepOperator{I}, site, bL::I, bR::I) where {I}
    c = letter.c
    V = _space(site)
    f1 = only(fusiontrees((bL, c), bR, (false, false)))
    f2 = only(fusiontrees((bR,), bR, (false,)))
    κ = zeros(ComplexF64, Vect[I](bL => 1) ⊗ Vect[I](c => 1) ← Vect[I](bR => 1))
    _add_coupler_coeff!(κ, f1, f2, 1, 1, ComplexF64(1))
    blk = zeros(ComplexF64, Vect[I](bL => 1) ⊗ V ← V ⊗ Vect[I](bR => 1))
    _add_bond_entry!(blk, instantiate(letter, site), κ)
    return blk
end

"""
    irrep_mpo_tensors(Ws, bondsectors, lat) -> Vector{<:AbstractTensorMap}

Assemble the symmetric MPO from the reduced bond matrices and bond sectors (from [`mpo`](@ref)): site
tensors `W_i : B_{i-1} ⊗ V_i ← V_i ⊗ B_i` (MPSKit convention) with one-dimensional boundary bonds. Each
reduced entry `(l, r, letter, coeff)` places the letter's tensor into the `(l → r)` transition weighted by
`coeff`, coupling the operator charge into the bond via the fusion `b_L ⊗ c → b_R`.
"""
function irrep_mpo_tensors(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}},
        bondsectors::Vector{Vector{I}}, sites
    ) where {I}
    # `bsecs[i]` is the bond to the *left* of site i; `bsecs[1]` is the trivial left boundary.
    return _mpo_tensors(Ws, vcat([I[unit(I)]], bondsectors), _tolattice(sites))
end

# Shared assembly: `bsecs[i]` is the bond to the *left* of site i (length N+1), so the finite and
# infinite paths differ only in what `bsecs[1]` is.
function _mpo_tensors(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}},
        bsecs::Vector{Vector{I}}, sites
    ) where {I}
    N = length(Ws)
    bspaces = map(_bond_space, bsecs)
    bdegs = map(_deg_indices, bsecs)

    return map(1:N) do i
        site = sites[i]
        V = _space(site)
        secL, secR = bsecs[i], bsecs[i + 1]
        Bleft, Bright = bspaces[i], bspaces[i + 1]
        degL, degR = bdegs[i], bdegs[i + 1]

        W = zeros(ComplexF64, Bleft ⊗ V ← V ⊗ Bright)

        # Per *letter*, not per entry: κ collects every coefficient carrying that letter, so this is
        # `O(d²)` contractions rather than `O(D_L · D_R)`.
        ops = Dictionary{IrrepOperator{I}, Any}()
        couplers = Dictionary{IrrepOperator{I}, Any}()
        f1s = Dictionary{Tuple{I, I, I}, Any}()
        f2s = Dictionary{I, Any}()

        for (idx, localop) in storedpairs(Ws[i])
            l, r = Tuple(idx)
            bL, dL = secL[l], degL[l]
            bR, dR = secR[r], degR[r]
            for (letter, coeff) in pairs(localop)
                c = letter.c
                get!(() -> instantiate(letter, site), ops, letter)
                # Fusion `(bL, c) → bR`: running bond FIRST, matching the caterpillar. Charge-first
                # flips the sign at antisymmetric inner vertices (1⊗1→1, so K ≥ 3).
                κ = get!(couplers, letter) do
                    return zeros(ComplexF64, Bleft ⊗ Vect[I](c => 1) ← Bright)
                end
                f1 = get!(() -> only(fusiontrees((bL, c), bR, (false, false))), f1s, (bL, c, bR))
                f2 = get!(() -> only(fusiontrees((bR,), bR, (false,))), f2s, bR)
                _add_coupler_coeff!(κ, f1, f2, dL, dR, coeff)
            end
        end

        for (letter, κ) in pairs(couplers)
            _add_bond_entry!(W, ops[letter], κ)
        end
        return W
    end
end

"""
    islossless(H::OperatorSum[, alg]) -> Bool

Whether the compressed MPO reproduces `H` exactly: [`mpo_terms`](@ref) of the MPO compared with
[`chain_terms`](@ref)`(H)` by `≈` (term *set* exactly, coefficients approximately). Purely symbolic, so
cheap and valid for fermionic sectors. Finite chains only. After a truncating [`SVDBondAlgorithm`](@ref)
`false` is expected.
"""
function islossless(H::OperatorSum{I, <:FiniteChain}, args...) where {I}
    return mpo_terms(mpo(H, args...)) ≈ chain_terms(H)
end

islossless(::OperatorSum{I, <:InfiniteChain}, args...) where {I} = throw(
    ArgumentError("islossless is not supported on an infinite chain")
)

"""
    mpo_tensormap(Ts::AbstractVector) -> AbstractTensorMap

Contract a chain of MPO site tensors ([`irrep_mpo_tensors`](@ref), or [`jordan_mpo_tensors`](@ref) after
`map(TensorMap, ·)`) into the `N`-site operator in [`instantiate`](@ref)'s leg convention: codomain
`o₁…o_N`, domain `i₁…i_N` and the trailing total-charge leg; the trivial left boundary is dropped.

The tensor-level oracle `mpo_tensormap(irrep_mpo_tensors(mpo(H))) ≈ instantiate(H)`. Exponential in `N`,
but stays inside TensorKit, so it is valid for fermionic sectors.
"""
function mpo_tensormap(Ts::AbstractVector)
    N = length(Ts)
    N >= 1 || throw(ArgumentError("mpo_tensormap: need at least one site tensor"))
    net = [[i == 1 ? -(2N + 1) : i - 1, -i, -(N + i), i == N ? -(2N + 2) : i] for i in 1:N]
    O = removeunit(ncon(Ts, net), 2N + 1)    # legs o₁…o_N, i₁…i_N, bL, bR; drop the left boundary
    return permute(O, (ntuple(identity, N), (ntuple(i -> N + i, N)..., 2N + 1)))
end
