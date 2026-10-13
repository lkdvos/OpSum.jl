# The symbolic half of `jordan_mpo_tensors`: the bond matrices of a `SymbolicMPO` in Jordan order, with
# nothing instantiated. This is the form the Kronecker power of the MPO tensor (`kronpower.jl`) starts from.

using SparseArrays: SparseMatrixCSC, sparse
using TensorKit: Sector

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

# Throws where `jordan_mpo_tensors` does (no identity channels known, non-trivial total charge, empty bond).
function _jordan_layout(W::SymbolicMPO{I, L}) where {I, L}
    pos, jsec = _jordan_bonds(W)
    N = length(W)
    Ws = map(1:N) do i
        nrows, ncols = length(jsec[i]), length(jsec[i + 1])
        entries = _jordan_entries(W.Ws[i], pos[i], pos[i + 1], nrows, ncols, i)
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
