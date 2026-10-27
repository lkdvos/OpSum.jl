# Jordan-form emission: one `SparseBlockTensorMap` per site, bond indices reordered (start, rest, finish).
# Both identity channels are padded in at every internal bond where the cover spent no index on them.

using BlockTensorKit: BlockTensorKit, SparseBlockTensorMap, SumSpace, eachspace
using SparseArrays: SparseMatrixCSC, sparse
using TensorKit: BraidingTensor, Vect, Sector, unit, tensormaptype

"""
    jordan_mpo_tensors(H::OperatorSum[, alg]) -> Vector{<:SparseBlockTensorMap}

`jordan_mpo_tensors(mpo(H, alg))`: the compressed MPO in **Jordan form**, one `SparseBlockTensorMap` per
site (of the chain or the unit cell), `W_i : B_{i-1} ⊗ V_i ← V_i ⊗ B_i`, with bond indices ordered
`(start channel, everything else, finish channel)` and identities at `(1, 1)` and `(end, end)` emitted as
`BraidingTensor`s. On a finite chain `H` must have total charge `unit(I)`. This is the shape MPSKit's
`JordanMPOTensor` consumes; use [`mpo_tensors`](@ref) for dense `TensorMap`s.
A truncating SVD [`BondStrategy`](@ref) that empties a bond is rejected.
"""
jordan_mpo_tensors(H::OperatorSum, args...) = jordan_mpo_tensors(mpo(H, args...))
jordan_mpo_tensors(W::SymbolicMPO) = _jordan_tensors(jordan_layout(W), W.lattice)

# Reorder one bond; `s`/`f` are the start/finish indices (`0` if absent); slots `1`/`end` are reserved for
# them. Returns `(pos, sec)`: `pos[old] -> new`, `sec[new]` the charge.
function _jordan_bond(sec::Vector{I}, s::Int, f::Int) where {I}
    n = length(sec)
    iszero(n) && throw(
        ArgumentError(
            "a bond of the reduced MPO is zero-dimensional, so it cannot carry the Jordan identity " *
                "corners; this only happens under a truncating SVD strategy"
        )
    )
    (iszero(s) || sec[s] == unit(I)) ||
        invariant_violation("the start channel carries the non-trivial charge $(sec[s])")
    (iszero(f) || sec[f] == unit(I)) ||
        invariant_violation("the finish channel carries the non-trivial charge $(sec[f])")
    (iszero(s) || s != f) || invariant_violation("the start and finish channels are the same bond index")

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

# `pos[i]` maps the old indices of the bond left of site `i` to Jordan slots, `jsec[i]` their charges
# (`N + 1` bonds). The finite boundary bonds stay one-dimensional: a pure start, a pure finish.
function _jordan_bonds(W::SymbolicMPO{I, <:FiniteChain}) where {I}
    isempty(W.Ws) && throw(
        ArgumentError("cannot emit a Jordan MPO for an empty operator: it has no bond structure")
    )
    N = length(W)
    W.bondsectors[N] == I[unit(I)] || throw(
        ArgumentError(
            "the right boundary bond is $(W.bondsectors[N]), not the trivial `[$(unit(I))]`: only a " *
                "total-charge-`unit` operator has a Jordan MPO"
        )
    )
    bonds = [_jordan_bond(W.bondsectors[i], W.start[i], W.finish[i]) for i in 1:(N - 1)]
    pos = Vector{Int}[Int[1], first.(bonds)..., Int[1]]
    jsec = Vector{I}[I[unit(I)], last.(bonds)..., I[unit(I)]]
    return pos, jsec
end

# The unit cell: every bond carries both identity channels; the bond left of site 1 is bond `L`.
function _jordan_bonds(W::SymbolicMPO{I, <:InfiniteChain}) where {I}
    L = length(W)
    bonds = [_jordan_bond(W.bondsectors[j], W.start[j], W.finish[j]) for j in 1:L]
    pos = Vector{Int}[first(bonds[mod1(j - 1, L)]) for j in 1:(L + 1)]
    jsec = Vector{I}[last(bonds[mod1(j - 1, L)]) for j in 1:(L + 1)]
    return pos, jsec
end

"""
    JordanLayout{I}

A [`SymbolicMPO`](@ref) permuted into Jordan form (see [`jordan_mpo_tensors`](@ref)). There are `N + 1`
bonds, bond `i` being the one *left* of site `i` (on an infinite chain bond `N + 1` is bond `1` again), so
`Ws[i]` maps the indices of bond `i` to those of bond `i + 1`:

  - `bondsectors[i]`: the charge of each index of bond `i`;
  - `start[i]` / `finish[i]`: the index of the start / finish identity channel of bond `i`, `0` if the
    bond has none. An internal or infinite bond has both, always at `1` and at its last index (padded
    where the compression spent none); the left boundary of a finite chain is a pure start (`1`, `0`) and
    its right boundary a pure finish (`0`, `1`), both one-dimensional of charge `unit`.

The identity sits at the corners `(1, 1)` and `(end, end)` of every `Ws[i]`.
"""
struct JordanLayout{I <: Sector}
    Ws::Vector{SparseMatrixCSC{SiteOperator{I}, Int}}
    bondsectors::Vector{Vector{I}}
    start::Vector{Int}
    finish::Vector{Int}
end

Base.length(J::JordanLayout) = length(J.Ws)

# Steps 1-2 of the emission, symbolic: the entries of every site in Jordan order, the structural invariants
# checked and the identity corners padded where missing. Throws where `jordan_mpo_tensors` does (non-trivial
# total charge, empty bond).
function jordan_layout(W::SymbolicMPO{I, L}) where {I, L}
    pos, jsec = _jordan_bonds(W)
    N = length(W)
    Ws = map(1:N) do i
        nrows, ncols = length(jsec[i]), length(jsec[i + 1])
        posL, posR = pos[i], pos[i + 1]
        # 1. permute the reduced entries into Jordan order (a bijection)
        entries = Dictionary{Tuple{Int, Int}, SiteOperator{I}}()
        for (idx, localop) in storedpairs(W.Ws[i])
            l, r = Tuple(idx)
            isempty(localop) && continue
            insert!(entries, (posL[l], posR[r]), localop)
        end

        # 2. the structural invariants, then the identity corners (padded where missing)
        for (row, col) in keys(entries)
            (ncols > 1 && col == 1 && row != 1) &&
                invariant_violation("row $row of site $i enters the Jordan start channel")
            (nrows > 1 && row == nrows && col != ncols) &&
                invariant_violation("the Jordan finish channel of site $i emits into column $col")
        end
        for (row, col, needed) in ((1, 1, ncols > 1), (nrows, ncols, nrows > 1))
            needed || continue
            op = get(entries, (row, col), nothing)
            if op === nothing
                insert!(entries, (row, col), one(SiteOperator{I}))
            elseif !isone(op)
                invariant_violation("the Jordan corner ($row, $col) of site $i is not the identity")
            end
        end
        rows = Int[first(k) for k in keys(entries)]
        cols = Int[last(k) for k in keys(entries)]
        return sparse(rows, cols, collect(entries), nrows, ncols)
    end
    ends = map(length, jsec)
    start = fill(1, N + 1)
    finish = copy(ends)
    if L <: FiniteChain
        finish[1] = 0
        start[N + 1] = 0
        finish[N + 1] = 1
    end
    return JordanLayout{I}(Ws, jsec, start, finish)
end

# Step 3 of the emission: the tensors of a layout.
function _jordan_tensors(layout::JordanLayout{I}, sites) where {I}
    N = length(layout)
    S = typeof(space(first(sites)))
    TT = AbstractTensorMap{ComplexF64, S, 2, 2}     # abstract: `BraidingTensor`s stay braidings
    spaces = map(sec -> SumSpace([Vect[I](c => 1) for c in sec]), layout.bondsectors)

    return map(1:N) do i
        site = sites[i]
        V = space(site)
        secL, secR = layout.bondsectors[i], layout.bondsectors[i + 1]

        # a unit-coefficient pass-through on a charge-preserving slot is a braiding
        W = SparseBlockTensorMap{TT}(undef, spaces[i] ⊗ V ← V ⊗ spaces[i + 1])
        blocks = Dictionary{Tuple{Letter{I}, I, I}, tensormaptype(S, 2, 2, ComplexF64)}()
        for (idx, op) in storedpairs(layout.Ws[i])
            row, col = Tuple(idx)
            bL, bR = secL[row], secR[col]
            if isone(op)
                bL == bR ||
                    invariant_violation("a pass-through bond entry connects charge $bL to charge $bR")
                W[row, 1, 1, col] = BraidingTensor{ComplexF64, S, Vector{ComplexF64}}(
                    eachspace(W)[row, 1, 1, col]
                )
                continue
            end
            (letter, coeff), rest = Iterators.peel(pairs(op))
            blk = coeff * get!(() -> _letter_block(letter, site, bL, bR), blocks, (letter, bL, bR))
            for (letter, coeff) in rest
                B = get!(() -> _letter_block(letter, site, bL, bR), blocks, (letter, bL, bR))
                add!(blk, B, coeff)
            end
            W[row, 1, 1, col] = blk
        end
        return W
    end
end
