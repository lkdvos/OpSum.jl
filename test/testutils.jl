# Helpers shared by the test files
# ================================
# These were copy-pasted across the suite (`physmatrix` into four files, `densedim` into three,
# `islossless` into two, `LO`/`onlyterm` into several). They live here so a change to the API only
# has to be made once.
#
# This file is *not* a test file: `test/runtests.jl` excludes it from the discovered suite. Each
# test file `include`s it, so running a file directly (`julia --project=test test/test_irrep_mpo.jl`,
# the workflow CLAUDE.md documents) still works.

using OpSum
using OpSum: irrep_mpo, mpo_terms, TermSum, OnsiteOp
using TensorKit: dim

# Wrap a bare alphabet letter as an `OnsiteOp`, for comparing against reduced bond-matrix entries.
LO(x) = OnsiteOp(x)

# The single `(TermKey, coeff)` pair of a one-term `TermSum`.
onlyterm(ts::TermSum) = only(pairs(ts.terms))

# Densify a `TensorMap` operator to a matrix: drop the dim-1 (boundary / charge) legs, keep the 2N
# physical axes ordered [out_1..N, in_1..N], and reorder to the kron index convention.
function physmatrix(t, N, d)
    A = convert(Array, t)
    A = dropdims(A; dims = Tuple(findall(==(1), size(A))))
    @assert ndims(A) == 2N
    perm = (reverse(1:N)..., reverse((N + 1):(2N))...)
    return reshape(permutedims(A, perm), d^N, d^N)
end

# Dense-equivalent bond dimension at bond `b`: the qdim-weighted sum of the per-sector
# multiplicities, i.e. what an unsymmetric MPO would have to carry.
densedim(secs, b) = sum(dim(c) for c in secs[b])

# Faithfulness: the compressed MPO reconstructs the original term-sum exactly. This is the primary
# correctness oracle — cheap, independent of `N`, and safe for fermionic sectors (unlike densifying).
# Says nothing after a *truncating* `SVDBondAlgorithm`, which is lossy by construction.
function islossless(H::TermSum, sites, alg...)
    Ws, secs = irrep_mpo(H, sites, alg...)
    back = mpo_terms(Ws, secs)
    Set(keys(back.terms)) == Set(keys(H.terms)) || return false
    return all(back.terms[k] ≈ H.terms[k] for k in keys(H.terms))
end
