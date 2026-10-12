# Helpers shared by the test files (not a test file: runtests.jl excludes it; each test file includes it).

using OpSum
using OpSum: Term, Terms, SiteOperator, ops, tree, total, arity
using TensorKit: dim

# The reduced data of an `mpo(H)`: `Ws, secs = reduced(mpo(H))`.
reduced(W::SymbolicMPO) = (W.Ws, W.bondsectors)

# Wrap a bare alphabet letter as a `SiteOperator`, for comparing against reduced bond-matrix entries.
LO(x) = SiteOperator(x)

# The single `Term` of a one-term operator.
onlyterm(ts::Terms) = only(ts)

# Densify a `TensorMap` operator to a matrix: drop the dim-1 (boundary / charge) legs, keep the 2N
# physical axes ordered [out_1..N, in_1..N], and reorder to the kron index convention.
function physmatrix(t, N, d)
    A = convert(Array, t)
    A = dropdims(A; dims = Tuple(findall(==(1), size(A))))
    @assert ndims(A) == 2N
    perm = (reverse(1:N)..., reverse((N + 1):(2N))...)
    return reshape(permutedims(A, perm), d^N, d^N)
end

# A `V ← V` operator from the `V ← V ⊗ unit` form `instantiate` gives.
plain(t) = numind(t) == 2 * numout(t) ? t : removeunit(t, numind(t))

# Dense-equivalent bond dimension at bond `b`: the qdim-weighted sum of the per-sector multiplicities.
densedim(secs, b) = sum(dim(c) for c in secs[b])

# Bases and letters for tests of the term algebra and the sweep (charged letters on a given space).

using TensorKit: TensorKit, ElementarySpace, Sector, Vect, sectortype, block, dual, unit, sectors,
    isomorphism, numind, numout, removeunit, ⊗, ←

# The complete basis of matrix units on `V`: the identity, then every matrix unit except the first of
# trivial charge (dependent on the identity), so a fermion's `n` is one letter. Memoised per space.
const _UNITBASES = Dict{Any, Any}()
function unitbasis(V::ElementarySpace)
    return get!(_UNITBASES, V) do
        I = sectortype(V)
        idel = isomorphism(ComplexF64, V ← V ⊗ Vect[I](unit(I) => 1))
        units = [
            instantiate(mu, V) for mu in instances(OpSum.IrrepOperator, V)
                if !(mu.c == unit(I) && mu.n <= 2)    # n = 1 is the identity itself
        ]
        OpSum._basis_from_elements(V, [idel; units])
    end
end

# The `n`-th element of a basis as a symbolic operator with coefficient one.
element(B::Basis, n::Integer) = SiteOperator(B.letters[n])

# The matrix unit `|out⟩⟨in|` (one-dimensional, multiplicity-one sectors) in the basis `B`.
function matrixunit(B::Basis, out::I, in::I) where {I <: Sector}
    V = TensorKit.space(B)
    c = only(⊗(out, dual(in)))
    t = zeros(ComplexF64, V ← V ⊗ Vect[I](c => 1))
    block(t, out) .= 1
    return project(t, B)
end

# The SU(2) spin vector on a single-irrep SU(2) basis, normalised so that `dot(S[i], S[j])` is the
# Cartesian `S⃗ᵢ · S⃗ⱼ`: √(s(s+1)(2s+1)) times the spin-1 matrix unit.
function spinvector(B::Basis)
    V = TensorKit.space(B)
    s = only(sectors(V)).j
    mu = instantiate(OpSum.IrrepOperator(TensorKit.SU2Irrep(1), 1), V)   # the spin-1 matrix unit
    return sqrt(s * (s + 1) * (2s + 1)) * project(mu, B)
end

# `(; Sp, Sm, Sz)` of a U(1)-graded spin site from its sectors in descending `m`, in the basis `B`.
function spinladder(B::Basis, sectors::AbstractVector)
    n = length(sectors)
    s = (n - 1) / 2
    m(k) = s - (k - 1)
    ladder(k) = sqrt(s * (s + 1) - m(k + 1) * (m(k + 1) + 1))
    Sz = sum(m(k) * matrixunit(B, sectors[k], sectors[k]) for k in 1:n if !iszero(m(k)))
    Sp = sum(ladder(k) * matrixunit(B, sectors[k], sectors[k + 1]) for k in 1:(n - 1))
    Sm = sum(ladder(k) * matrixunit(B, sectors[k + 1], sectors[k]) for k in 1:(n - 1))
    return (; Sp, Sm, Sz)
end

# `(; c, cd, n)` of a spinless fermionic mode between the sectors `vac` and `occ`, in the basis `B`.
fermionmode(B::Basis, vac, occ) =
    (; c = matrixunit(B, vac, occ), cd = matrixunit(B, occ, vac), n = matrixunit(B, occ, occ))
