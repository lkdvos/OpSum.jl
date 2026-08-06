# Jordan-form MPO emission: one `SparseBlockTensorMap` per site, virtual legs `SumSpace`s with one
# level per bond index, reordered into upper-triangular (start channel, rest, finish channel) form —
# what MPSKit's `JordanMPOTensor` / `FiniteMPOHamiltonian` consume. `irrep_mpo_tensors` (irrepmpo.jl)
# emits the dense-bond shape instead.
#
# THE JORDAN PADDING TRADE. Both identity channels are emitted at every internal bond even where the
# cover spent no index on them, so the result is minimal *among Jordan-form MPOs* and at most `+2` per
# bond over `irrep_mpo`'s unconstrained minimum (in practice `+1` at the first internal bond and `+1`
# at the last). Padded channels are pure identity chains and cannot change the operator: the padded
# start is reachable only from the left boundary and emits nothing into the finish, and the padded
# finish chain is reachable from nothing. The one thing that *does* touch the cover is `_force_finish!`
# (irrepgraph.jl), so the finish channel emits `1 · id` rather than a weighted letter.

using BlockTensorKit: BlockTensorKit, SparseBlockTensorMap, SumSpace, eachspace
using TensorKit: BraidingTensor, Vect, ElementarySpace, fusiontrees, unit, isomorphism,
    tensormaptype, @tensor
using .IrrepTensorOperators: IrrepOperator

"""
    jordan_mpo_tensors(H::TermSum[, alg]) -> Vector{<:SparseBlockTensorMap}

Compress `H` into a reduced MPO (as [`irrep_mpo`](@ref)) and emit it in **Jordan form**: one
`BlockTensorKit.SparseBlockTensorMap` per site, `W_i : B_{i-1} ⊗ V_i ← V_i ⊗ B_i`, whose virtual legs
are `SumSpace`s carrying one level `Vect[I](charge => 1)` per bond index, ordered as

    (start channel, everything else, finish channel)

with the identity at `(1, 1)` and `(end, end)`. The boundary bonds `B_0` and `B_N` are
one-dimensional and trivially charged, so `H` must have total charge `unit(I)`.

Diagonal pass-through entries with unit coefficient are emitted as `TensorKit.BraidingTensor`s, which
is what lets a consumer store them as scalars rather than dense blocks.

This is the shape MPSKit's `JordanMPOTensor` / `FiniteMPOHamiltonian` consume; OpSum does not depend
on MPSKit, and nothing here does more than name the convention. Use [`irrep_mpo_tensors`](@ref) for
the dense per-site `TensorMap`s instead.

`alg` defaults to [`BipartiteAlgorithm`](@ref). An [`SVDBondAlgorithm`](@ref) bond basis is a
*mixture* of prefix states in which neither identity channel is a basis vector, so both are padded at
every bond; a truncation aggressive enough to empty a bond is rejected, since a Jordan-form MPO with a
zero-dimensional bond cannot carry its identity corners.
"""
function jordan_mpo_tensors(
        H::TermSum, alg::Union{BipartiteAlgorithm, SVDBondAlgorithm} = BipartiteAlgorithm()
    )
    tt = ITOTermTable(H)
    N = nvertices(tt)
    Ws, bondsectors, starts, finishes = _irrep_channels(tt, N, bondstrategy(alg))
    isempty(Ws) && throw(
        ArgumentError("cannot emit a Jordan MPO for an empty `TermSum`: it has no bond structure")
    )
    return jordan_mpo_tensors(Ws, bondsectors, starts, finishes, lattice(H))
end

# Reorder one internal bond; `s`/`f` are the start/finish indices (`0` if the cover spent none).
# Slots `1`/`end` are *reserved* for them live or not, so padding needs no separate bookkeeping.
# Returns `(pos, sec)`: `pos[old] -> new`, `sec[new]` the new charge.
function _jordan_bond(sec::Vector{I}, s::Int, f::Int) where {I}
    n = length(sec)
    iszero(n) && throw(
        ArgumentError(
            "a bond of the reduced MPO is zero-dimensional, so it cannot carry the Jordan identity " *
                "corners; this only happens under a truncating `SVDBondAlgorithm`"
        )
    )
    (iszero(s) || sec[s] == unit(I)) ||
        _invariant("the start channel carries the non-trivial charge $(sec[s])")
    (iszero(f) || sec[f] == unit(I)) ||
        _invariant("the finish channel carries the non-trivial charge $(sec[f])")
    (iszero(s) || s != f) || _invariant("the start and finish channels are the same bond index")

    pos = Vector{Int}(undef, n)
    newsec = I[unit(I)]                       # slot 1 — the start channel, live or padded
    iszero(s) || (pos[s] = 1)
    for j in 1:n
        (j == s || j == f) && continue
        push!(newsec, sec[j])
        pos[j] = length(newsec)
    end
    push!(newsec, unit(I))                    # slot end — the finish channel, live or padded
    iszero(f) || (pos[f] = length(newsec))
    return pos, newsec
end

# One letter's site tensor, fusing `(bL, c) → bR` (running bond first, as in `irrep_mpo_tensors`).
# Memoised per `(letter, bL, bR)`, so the trees and the contraction are paid once per distinct triple.
function _jordan_letter_block(letter::IrrepOperator{I}, V, bL::I, bR::I) where {I}
    c = letter.c
    Vc = Vect[I](c => 1)
    # pass-through (c = unit) keeps the trivial charge leg so the bond contraction is uniform
    O = ispassthrough(letter) ? isomorphism(ComplexF64, V ← V ⊗ Vc) : instantiate(letter, V)
    κ = zeros(ComplexF64, Vect[I](bL => 1) ⊗ Vc ← Vect[I](bR => 1))
    f1 = only(fusiontrees((bL, c), bR, (false, false)))
    f2 = only(fusiontrees((bR,), bR, (false,)))
    κ[f1, f2][1, 1, 1] = 1
    @tensor blk[bl o; ii br] := O[o; ii cc] * κ[bl cc; br]
    return blk
end

"""
    jordan_mpo_tensors(Ws, bondsectors, starts, finishes, sites) -> Vector{<:SparseBlockTensorMap}

The reduced-data form of [`jordan_mpo_tensors`](@ref): `(Ws, bondsectors)` as `irrep_mpo` returns
them, plus the per-bond start/finish channel indices from `_irrep_channels` (`0` where the cover did
not spend an index — that channel is then padded).

Two structural invariants are checked per site rather than assumed, because a violation would be
silent: nothing may enter the start column except from the start row, and the finish row may emit into
nothing but the finish column. Both follow from how the two channels are read off the cover, but they
are the properties a consumer's block accessors rely on, and a wrong one drops operators on the floor
instead of erroring.
"""
function jordan_mpo_tensors(
        Ws::Vector{<:SparseMatrixCSC{SiteOperator{I}}}, bondsectors::Vector{Vector{I}},
        starts::Vector{Int}, finishes::Vector{Int}, sites
    ) where {I}
    N = length(Ws)
    N == length(sites) ||
        throw(ArgumentError("got $N bond matrices for $(length(sites)) sites"))
    bondsectors[N] == I[unit(I)] || throw(
        ArgumentError(
            "the right boundary bond is $(bondsectors[N]), not the trivial `[$(unit(I))]`: only a " *
                "total-charge-`unit` operator has a Jordan MPO"
        )
    )

    # `pos[i]` / `jsec[i]` describe the bond to the *left* of site i; both boundaries stay
    # one-dimensional (and are simultaneously that bond's start and finish channel).
    pos = Vector{Vector{Int}}(undef, N + 1)
    jsec = Vector{Vector{I}}(undef, N + 1)
    pos[1], jsec[1] = Int[1], I[unit(I)]
    pos[N + 1], jsec[N + 1] = Int[1], I[unit(I)]
    for i in 1:(N - 1)
        pos[i + 1], jsec[i + 1] = _jordan_bond(bondsectors[i], starts[i], finishes[i])
    end

    S = typeof(first(sites))
    TT = AbstractTensorMap{ComplexF64, S, 2, 2}     # abstract: `BraidingTensor`s stay braidings
    spaces = map(sec -> SumSpace([Vect[I](c => 1) for c in sec]), jsec)

    return map(1:N) do i
        V = sites[i]
        secL, secR = jsec[i], jsec[i + 1]
        nrows, ncols = length(secL), length(secR)

        # 1. permute the reduced entries into Jordan order. The permutation is a bijection, so no two
        #    reduced entries can land in the same slot.
        entries = Dictionary{Tuple{Int, Int}, SiteOperator{I}}()
        for (idx, localop) in storedpairs(Ws[i])
            l, r = Tuple(idx)
            op = prune(localop)
            isempty(op) && continue
            insert!(entries, (pos[i][l], pos[i + 1][r]), op)
        end

        # 2. the two structural invariants, then the two identity corners (padding them where the
        #    compression did not produce them — see the file header).
        for (row, col) in keys(entries)
            (ncols > 1 && col == 1 && row != 1) &&
                _invariant("row $row of site $i enters the Jordan start channel")
            (nrows > 1 && row == nrows && col != ncols) &&
                _invariant("the Jordan finish channel of site $i emits into column $col")
        end
        for corner in ((1, 1, ncols > 1), (nrows, ncols, nrows > 1))
            row, col, needed = corner
            needed || continue
            op = get(entries, (row, col), nothing)
            if op === nothing
                insert!(entries, (row, col), one(SiteOperator{I}))
            elseif !isone(op)
                _invariant("the Jordan corner ($row, $col) of site $i is not the identity")
            end
        end

        # 3. emit. A unit-coefficient pass-through on a charge-preserving slot *is* the braiding, and
        #    is emitted as one so that a consumer can keep it out of dense storage.
        W = SparseBlockTensorMap{TT}(undef, spaces[i] ⊗ V ← V ⊗ spaces[i + 1])
        blocks = Dictionary{Tuple{IrrepOperator{I}, I, I}, tensormaptype(S, 2, 2, ComplexF64)}()
        for ((row, col), op) in pairs(entries)
            bL, bR = secL[row], secR[col]
            if isone(op)
                bL == bR ||
                    _invariant("a pass-through bond entry connects charge $bL to charge $bR")
                W[row, 1, 1, col] = BraidingTensor{ComplexF64, S, Vector{ComplexF64}}(
                    eachspace(W)[row, 1, 1, col]
                )
                continue
            end
            blk = nothing
            for (letter, coeff) in pairs(op)
                B = get!(() -> _jordan_letter_block(letter, V, bL, bR), blocks, (letter, bL, bR))
                blk = blk === nothing ? coeff * B : blk + coeff * B
            end
            W[row, 1, 1, col] = blk
        end
        return W
    end
end
