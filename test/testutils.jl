# Helpers shared by the test files
# ================================
# These were copy-pasted across the suite (`physmatrix` into four files, `densedim` into three,
# `LO`/`onlyterm` into several). They live here so a change to the API only has to be made once.
# `islossless` and `mpo_tensormap` used to live here too; they are now OpSum's public API.
#
# This file is *not* a test file: `test/runtests.jl` excludes it from the discovered suite. Each
# test file `include`s it, so running a file directly (`julia --project=test test/test_irrep_mpo.jl`,
# the workflow CLAUDE.md documents) still works.

using OpSum
using OpSum: Term, Terms, TermSum, SiteOperator, ops, tree, total, arity
using TensorKit: dim

# Wrap a bare alphabet letter as a `SiteOperator`, for comparing against reduced bond-matrix entries.
LO(x) = SiteOperator(x)

# The single `Term` of a one-term bag or operator. `only` does this directly; the helper stays
# because it reads better at the call sites and it asserts the "exactly one" part by name.
onlyterm(ts::Terms) = only(ts)
onlyterm(H::TermSum) = only(H)

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
