# Reduced MPO for the ITO automaton: `irrep_mpo` compresses a `TermSum` into reduced bond matrices
# plus per-bond charge sectors, by running a per-bond-sector sweep (irrepgraph.jl, strategy chosen by
# `algorithms.jl`) over the flat `ITOTermTable`. `mpo_terms` inverts it at the reduced level
# (faithfulness) and `irrep_mpo_tensors` assembles the symmetric `TensorMap`s. Any arity K ≥ 0.

using SparseArrays: SparseMatrixCSC
using TensorKit: Vect, ElementarySpace, fusiontrees, permute, dim, unit, isomorphism, @tensor,
    ncon, removeunit
using .IrrepTensorOperators: IrrepOperator

"""
    irrep_mpo(H::TermSum[, alg]) -> (Ws, bondsectors)

Compress the ITO Hamiltonian `H` over the `N = length(lattice(H))` sites it is defined on into a
reduced MPO. The lattice travels with `H` (see [`opsum`](@ref)), which is also where every letter was
checked against the space of the site it acts on.

Returns `Ws::Vector{SparseMatrixCSC{SiteOperator{I}, Int}}` (one reduced bond
matrix per site; entries are ITO letters times reduced coefficients) and `bondsectors::Vector{
Vector{I}}` where `bondsectors[i]` gives the bond charge of each bond index to the right of site
`i` (so `size(Ws[i], 2) == length(bondsectors[i])`). The left boundary (bond 0) is a single
trivial-charge index.

`alg` is the algorithm selector: `BipartiteAlgorithm()` (the default, lossless minimum vertex cover)
or `SVDBondAlgorithm(trunc; sweep)`, whose `sweep` picks between the two truncation semantics
(`IndependentSVD`, the default and the historical behaviour, versus `SequentialSVD`). Each selector
names a [`BondStrategy`](@ref); the sweeps live in irrepgraph.jl.
"""
function irrep_mpo(H::TermSum, alg::Union{BipartiteAlgorithm, SVDBondAlgorithm} = BipartiteAlgorithm())
    tt = ITOTermTable(H)
    return _irrep_sweep(tt, nvertices(tt), bondstrategy(alg))
end

"""
    irrep_mpo(H, chain::InfiniteChain[, alg]) -> InfiniteMPO

Compress the ITO Hamiltonian of an *infinite* chain with a repeating unit cell of
`L = length(chain)` sites. `H` is the generating set: the operator represented is
`Σ_{n ∈ ℤ} translate(H, n·L)`, so each translation class must appear exactly once in `H` (see
[`unitcell_terms`](@ref)).

`H` is normally a latticeless [`Terms`](@ref) bag — straight out of `couple`, `dot` or `project` — for
the same reason `unitcell_terms` returns one: the chain already names the space of *every* site, by
wraparound, so there is no separate lattice to bind. A `TermSum` is accepted too, and its lattice is
then required to agree with the chain site by site. Either way the letters are checked against those
spaces when the window is built.

Returns an [`InfiniteMPO`](@ref): `L` reduced bond matrices and `L` bond-charge lists with bond `0`
identified with bond `L`, plus the indices of the two identity channels (the boundary vectors). It
destructures as `Ws, bondsectors = irrep_mpo(H, chain)`, matching the finite contract, and feeds
`irrep_mpo_tensors(H_inf, chain)` directly.

Only `BipartiteAlgorithm` (the default, lossless) is supported: `SVDBondAlgorithm` compresses each
bond of a finite chain independently against a hard-coded vacuum-terminated bond layout and has no
notion of a bond basis that closes on itself.
"""
function irrep_mpo(
        gen::Terms, chain::InfiniteChain, ::BipartiteAlgorithm = BipartiteAlgorithm()
    )
    return _infinite_window(unitcell_terms(gen, length(chain)), chain)
end

function irrep_mpo(
        H::TermSum{I}, chain::InfiniteChain, alg::BipartiteAlgorithm = BipartiteAlgorithm()
    ) where {I}
    _check_chain_lattice(H, chain)
    return irrep_mpo(Terms{I}(H.terms), chain, alg)
end

function irrep_mpo(::Union{Terms, TermSum}, ::InfiniteChain, ::SVDBondAlgorithm)
    throw(
        ArgumentError(
            "SVDBondAlgorithm does not extend to infinite chains: it compresses each bond " *
                "independently against a vacuum-terminated layout, with no bond basis that closes " *
                "on itself. Use BipartiteAlgorithm() (the default)."
        )
    )
end

"""
    irrep_mpo(H::MixedSum, chain::InfiniteChain[, alg]) -> InfiniteMPO
    irrep_mpo(H::MixedSum, sites[, alg]) -> (Ws, bondsectors)

Compress a Hamiltonian carrying **exponentially decaying** interactions alongside its finite-range
terms (see [`expterm`](@ref)). Each channel becomes a single bond index with `λ` on its diagonal, so
its cost is independent of the interaction range.

The lattice fixes the translation period of a channel: on an [`InfiniteChain`](@ref) of `L` sites its
entry and exit step by `L` (and `H` is a generating set, as for a plain `Terms`), while on a finite
`sites` vector every site is a possible entry, i.e. the operator represented is the whole geometric sum
truncated to the chain — `chain_terms(H, length(sites))` spells it out. Only `BipartiteAlgorithm` (the
default) is supported.

A `MixedSum` is latticeless, like the [`Terms`](@ref) bag it is built from, so the finite form takes
its `sites` explicitly.
"""
function irrep_mpo(
        H::MixedSum, chain::InfiniteChain, ::BipartiteAlgorithm = BipartiteAlgorithm()
    )
    return _infinite_window(unitcell_terms(H, length(chain)), chain)
end

function irrep_mpo(
        H::MixedSum, sites::AbstractVector{<:ElementarySpace},
        ::BipartiteAlgorithm = BipartiteAlgorithm()
    )
    N = length(sites)
    return _irrep_sweep(
        ITOTermTable(opsum(sites, H.terms)), N, VertexCover();
        channels = _lower_channels(H.channels, 1)
    )
end

# a Hamiltonian of nothing but channels
irrep_mpo(H::ExpSum, lattice) = irrep_mpo(MixedSum(H), lattice)
irrep_mpo(H::ExpSum, lattice, alg::BipartiteAlgorithm) = irrep_mpo(MixedSum(H), lattice, alg)
irrep_mpo(H::ExpSum, lattice, alg::SVDBondAlgorithm) = irrep_mpo(MixedSum(H), lattice, alg)

function irrep_mpo(::MixedSum, ::Any, ::SVDBondAlgorithm)
    throw(
        ArgumentError(
            "SVDBondAlgorithm cannot compress an exponentially decaying interaction: it needs the " *
                "whole bond coefficient matrix up front, and a geometric channel is a bond index " *
                "with a diagonal rather than a column of it. Use BipartiteAlgorithm() (the default)."
        )
    )
end

"""
    mpo_terms(Ws, bondsectors, sites; leftidx = 1, rightidx = nothing) -> TermSum
Reconstruct the operator generated by a reduced MPO (inverse of `irrep_mpo` at the reduced level):
enumerate every path through the bond matrices, multiply the reduced coefficients along it, and read
the active ITO letters (skipping pass-through) together with their outgoing bond charges — which are
exactly the caterpillar running bonds a term carries. `sites` is the lattice to hand back, since the
bond data names charges but not physical spaces. For the lossless bipartite compression this recovers
the original operator exactly: `mpo_terms(irrep_mpo(H)..., lattice(H)) ≈ H` is the faithfulness
check, which [`islossless`](@ref) packages up.

`leftidx` and `rightidx` are the boundary vectors. The finite defaults (`1`, and "accept any final
index", which is what a vacuum-terminated chain needs since its last bond is one-dimensional) become
the two identity channels for an infinite MPO tiled over a window — see `mpo_terms_window`.
"""
function mpo_terms(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}},
        bondsectors::Vector{Vector{I}},
        sites;
        leftidx::Int = 1, rightidx::Union{Nothing, Int} = nothing
    ) where {I}
    N = length(Ws)
    N == 0 && return opsum(sites)

    cols = Tuple{Vector{Int}, Vector{ITOKey{I}}, ComplexF64}[]
    function walk(i, leftidx, coeff, sites, keys)
        if i > N
            # `leftidx` here is the *running* bond index the path arrived on; a finite chain accepts
            # any final index, an infinite window only the done channel
            (rightidx === nothing || leftidx == rightidx) || return
            push!(cols, (copy(sites), copy(keys), coeff))
            return
        end
        for (idx, localop) in storedpairs(Ws[i])
            l, r = Tuple(idx)
            l == leftidx || continue
            for (letter, c) in pairs(localop)
                # the outgoing bond charge of an active (non-pass-through) letter is that position's
                # running caterpillar bond `bₖ`; idle sites contribute nothing
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

    return opsum(sites, (Term{I}(s, k, c) for (s, k, c) in cols))
end

# Site tensors `W_i : B_{i-1} ⊗ V_i ← V_i ⊗ B_i` (MPSKit leg convention), virtual legs built from the
# per-sector bond multiplicities.

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
    out = Vector{Int}(undef, length(sec))
    for (j, c) in enumerate(sec)
        counts[c] = get(counts, c, 0) + 1
        out[j] = counts[c]
    end
    return out
end

# Contract one letter's accumulated bond coupler into the site tensor. Split out so that `@tensor`
# specialises on the concrete tensor types even though the per-site memo tables below are
# heterogeneously typed.
function _add_bond_entry!(W, O, κ)
    @tensor W[bl o; ii br] += O[o; ii cc] * κ[bl cc; br]
    return W
end

# Write one reduced coefficient into a coupler block. Also a function barrier: this runs once per
# *stored bond entry* — `O(D_L · D_R)` of them, against `O(d²)` contractions above — so it is the loop
# that must not stay dynamically dispatched off the memo tables' `Any` element type.
function _add_coupler_coeff!(κ, f1, f2, dL::Int, dR::Int, coeff)
    κ[f1, f2][dL, 1, dR] += coeff
    return κ
end

"""
    irrep_mpo_tensors(Ws, bondsectors, sites) -> Vector{<:AbstractTensorMap}

Assemble the symmetric MPO from the reduced bond matrices + bond sectors (from `irrep_mpo`). Site
tensor `W_i : B_{i-1} ⊗ V_i ← V_i ⊗ B_i` (MPSKit convention); the boundary bonds `B_0`, `B_N` are
one-dimensional.
Each reduced entry `(l, r, letter, coeff)` places the ITO letter's tensor into the `(l → r)` bond
transition weighted by `coeff`, coupling the operator charge into the bond via the (forward)
fusion `b_L ⊗ c → b_R`.
"""
function irrep_mpo_tensors(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}},
        bondsectors::Vector{Vector{I}}, sites
    ) where {I}
    # Every internal bond is shared by two site tensors, so build its graded space and degeneracy
    # indices once rather than twice (as `Bright`/`degR` of site i and `Bleft`/`degL` of site i+1).
    # `bsecs[i]` is the bond to the *left* of site i; `bsecs[1]` is the trivial left boundary.
    N = length(Ws)
    bsecs = Vector{Vector{I}}(undef, N + 1)
    bsecs[1] = I[unit(I)]
    for i in 1:N
        bsecs[i + 1] = bondsectors[i]
    end
    return _mpo_tensors(Ws, bsecs, sites)
end

"""
    irrep_mpo_tensors(H::InfiniteMPO, lattice::InfiniteChain) -> Vector{<:AbstractTensorMap}

Assemble the `L` symmetric site tensors of an infinite MPO. Identical to the finite assembly except
that the bond to the left of site 1 is bond `L` rather than the vacuum, so the returned tensors tile:
`space(Ts[1], 1) == space(Ts[L], 4)'`.
"""
function irrep_mpo_tensors(H::InfiniteMPO{I}, lattice::InfiniteChain) where {I}
    L = length(H)
    length(lattice) == L ||
        throw(ArgumentError("unit cell length mismatch: MPO has $L sites, lattice has $(length(lattice))"))
    # the only change from the finite case: bond 0 *is* bond L (wrap-around instead of vacuum)
    bsecs = Vector{Vector{I}}(undef, L + 1)
    bsecs[1] = H.bondsectors[L]
    for j in 1:L
        bsecs[j + 1] = H.bondsectors[j]
    end
    return _mpo_tensors(H.Ws, bsecs, lattice)
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

    # `map` infers a concrete `Vector{<:AbstractTensorMap}` element type (all site tensors share
    # the same `B ⊗ V ← V ⊗ B` space *type*), unlike an untyped preallocated buffer.
    return map(1:N) do i
        V = sites[i]
        secL, secR = bsecs[i], bsecs[i + 1]
        Bleft, Bright = bspaces[i], bspaces[i + 1]
        degL, degR = bdegs[i], bdegs[i + 1]

        W = zeros(ComplexF64, Bleft ⊗ V ← V ⊗ Bright)

        # Per *letter*, not per entry: κ_letter collects every reduced coefficient carrying that
        # letter, so it is `O(d²)` contractions rather than `O(D_L · D_R)`.
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
                # V ← V ⊗ Vect[c]; pass-through (c = unit) carries a trivial charge leg so the
                # bond contraction is uniform (instantiate would drop it, returning bare id(V)).
                get!(ops, letter) do
                    Vc = Vect[I](c => 1)
                    return ispassthrough(letter) ? isomorphism(ComplexF64, V ← V ⊗ Vc) :
                        instantiate(letter, V)
                end
                # Fusion `(bL, c) → bR`: running bond FIRST, matching the caterpillar. Charge-first
                # flips the sign at antisymmetric inner vertices (1⊗1→1, so K ≥ 3).
                κ = get!(couplers, letter) do
                    return zeros(ComplexF64, Bleft ⊗ Vect[I](c => 1) ← Bright)
                end
                f1 = get!(() -> only(fusiontrees((bL, c), bR, (false, false))), f1s, (bL, c, bR))
                f2 = get!(() -> only(fusiontrees((bR,), bR, (false,))), f2s, bR)
                # `(l, r)` determines `(bL, dL, bR, dR)` and the letter fixes `c`, so distinct stored
                # entries always address distinct slots; the `+=` only matters if a `SiteOperator`
                # carries one letter twice, which `+` prevents.
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
    islossless(H::TermSum[, alg]) -> Bool

Whether the compressed MPO reproduces `H` exactly: reconstruct the operator the reduced MPO generates
with [`mpo_terms`](@ref) and compare it against `H` with `≈`, i.e. term *set* exactly and coefficients
approximately.

This is the primary correctness check for a construction — cheap, purely symbolic, independent of `N`,
and valid for fermionic sectors, where densifying is not a well-defined operation.

Only meaningful for lossless compression: after a truncating [`SVDBondAlgorithm`](@ref) `false` is the
expected answer rather than a bug.
"""
function islossless(H::TermSum, args...)
    Ws, secs = irrep_mpo(H, args...)
    return mpo_terms(Ws, secs, lattice(H)) ≈ H
end

"""
    mpo_tensormap(Ts::AbstractVector) -> AbstractTensorMap

Contract a chain of MPO site tensors — as [`irrep_mpo_tensors`](@ref) returns them, or
[`jordan_mpo_tensors`](@ref) after `map(TensorMap, ·)` — into the single `N`-site operator, in
[`instantiate`](@ref)'s leg convention: codomain `o₁…o_N`, domain `i₁…i_N` and the trailing
total-charge leg. The trivial left boundary bond is dropped.

The tensor-level oracle, `mpo_tensormap(irrep_mpo_tensors(irrep_mpo(H)..., lattice(H))) ≈
instantiate(H)`. Exponential in `N`, so small systems only — but it stays inside TensorKit and never
materialises a dense array, so unlike `convert(Array, ·)` it is valid for fermionic sectors.
"""
function mpo_tensormap(Ts::AbstractVector)
    N = length(Ts)
    N >= 1 || throw(ArgumentError("mpo_tensormap: need at least one site tensor"))
    net = [[i == 1 ? -(2N + 1) : i - 1, -i, -(N + i), i == N ? -(2N + 2) : i] for i in 1:N]
    O = ncon(Ts, net)                # legs: o₁…o_N, i₁…i_N, bL, bR
    O = removeunit(O, 2N + 1)        # drop the trivial left boundary
    return permute(O, (ntuple(identity, N), (ntuple(i -> N + i, N)..., 2N + 1)))
end
