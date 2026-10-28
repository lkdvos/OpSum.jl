# Helpers shared by the test files (not a test file: runtests.jl excludes it; each test file includes it).

using OpSum
using OpSum: Term, Terms, SiteOperator, letters, tree, total, arity
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

# The dense bond dimensions of an MPO-like object (`SymbolicMPO`, `TaylorMPO`, ...), one per bond.
bonddims(T) = [densedim(T.bondsectors, b) for b in eachindex(T.bondsectors)]

# The dense operator of an `OperatorSum` on a finite chain, as a `V ← V` map.
densify(H) = plain(instantiate(H))

# Bases and letters for tests of the term algebra and the sweep (charged letters on a given space).

using TensorKit: TensorKit, ElementarySpace, Sector, Vect, sectortype, block, dual, unit, sectors,
    isomorphism, numind, numout, removeunit, ⊗, ←, Trivial, U1Irrep, @tensor
using OpSum: opsum, expterm, mpo_tensormap, bond_space, deg_indices

# The complete basis of matrix units on `V`: the identity, then every matrix unit except the first of
# trivial charge (dependent on the identity), so a fermion's `n` is one letter. Memoised per space.
const _UNITBASES = Dict{Any, Any}()
function unitbasis(V::ElementarySpace)
    return get!(_UNITBASES, V) do
        I = sectortype(V)
        idel = isomorphism(ComplexF64, V ← V ⊗ Vect[I](unit(I) => 1))
        units = [
            OpSum.matrixunit(mu, V) for mu in OpSum.matrixunits(V)
                if !(mu.c == unit(I) && mu.n <= 2)    # n = 1 is the identity itself
        ]
        OpSum.basis_from_elements(V, [idel; units])
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
    mu = OpSum.matrixunit(OpSum.Letter(TensorKit.SU2Irrep(1), 1), V)   # the spin-1 matrix unit
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

# Test models, `name => H`, built on demand: every model compiles its own operator-sum code on first use, so only the
# requested ones are constructed. `finite_models()` gives all of them, `finite_models(names...)` the subset (in the given
# order); likewise `infinite_models`. Dense oracles (`H^N`, `exp(τH)`) need short chains; XXZ and J1-J2 have 5 sites so
# that site 3 is a bulk site for the taylor_mpo tables.
function _select_models(builders, names)
    isempty(names) && return [name => build() for (name, build) in builders]
    return [name => only(build() for (n, build) in builders if n == name) for name in names]
end

function finite_models(names::AbstractString...)
    xxz(U, i, j) = 0.5 * (U.S_plus_S_min[i, j] + U.S_min_S_plus[i, j]) + 0.8 * U.S_z_S_z[i, j]
    builders = [
        "Ising (Trivial)" => function ()
            S = spin_ops(Trivial)
            return opsum(
                FiniteChain(spin_basis(Trivial), 4),
                (-S.S_z_S_z[i, i + 1] for i in 1:3), (0.7 * S.S_x[i] for i in 1:4)
            )
        end,
        "XXZ (U1)" => function ()
            U = spin_ops(U1Irrep)
            return opsum(
                FiniteChain(spin_basis(U1Irrep), 5), (xxz(U, i, i + 1) for i in 1:4), (0.3 * U.S_z[i] for i in 1:5)
            )
        end,
        # the bulk tables and bond-dimension counts are read at site 3, with two sites to its right
        "J1-J2 (U1)" => function ()
            U = spin_ops(U1Irrep)
            return opsum(
                FiniteChain(spin_basis(U1Irrep), 5), (xxz(U, i, i + 1) for i in 1:4), (0.4 * xxz(U, i, i + 2) for i in 1:3)
            )
        end,
        "t-V fermions (U1)" => function ()
            F1 = fermion_ops(U1Irrep)
            return opsum(
                FiniteChain(fermion_basis(U1Irrep), 4),
                (F1.f_plus_f_min[i, i + 1] + F1.f_plus_f_min[i + 1, i] + 2 * F1.f_num[i] * F1.f_num[i + 1] for i in 1:3)
            )
        end,
        "t-V fermions (parity), next-nearest hop" => function ()
            F0 = fermion_ops()
            return opsum(
                FiniteChain(fermion_basis(), 4),
                (F0.f_plus_f_min[i, i + 1] + F0.f_plus_f_min[i + 1, i] + 2 * F0.f_num[i] * F0.f_num[i + 1] for i in 1:3),
                (0.3 * (F0.f_plus_f_min[i, i + 2] + F0.f_plus_f_min[i + 2, i]) for i in 1:2)
            )
        end,
        "pairing (parity)" => function ()
            F0 = fermion_ops()
            return opsum(
                FiniteChain(fermion_basis(), 3),
                (F0.f_plus_f_plus[i, i + 1] - F0.f_min_f_min[i, i + 1] + 0.5 * F0.f_num[i] for i in 1:2)
            )
        end,
        "Hubbard (U1 x U1)" => function ()
            Hb = hubbard_ops(U1Irrep, U1Irrep)
            return opsum(
                FiniteChain(hubbard_basis(U1Irrep, U1Irrep), 3),
                (
                    Hb.u_plus_u_min[i, i + 1] + Hb.u_min_u_plus[i, i + 1] +
                        Hb.d_plus_d_min[i, i + 1] + Hb.d_min_d_plus[i, i + 1] for i in 1:2
                ),
                (4.0 * Hb.ud_num[i] for i in 1:3)
            )
        end,
    ]
    return _select_models(builders, names)
end

# Infinite-chain test models, `name => (H, cells)` with the number of unit cells of a window that is long enough
# for the dense oracles.
function infinite_models(names::AbstractString...)
    xxz(U, i, j; Δ = 1.0) = 0.5 * (U.S_plus_S_min[i, j] + U.S_min_S_plus[i, j]) + Δ * U.S_z_S_z[i, j]
    tV(F1, i, j; V = 2.0) = F1.f_plus_f_min[i, j] + F1.f_plus_f_min[j, i] + V * F1.f_num[i] * F1.f_num[j]
    builders = [
        "Heisenberg (U1)" => function ()
            U = spin_ops(U1Irrep)
            return (opsum(InfiniteChain(spin_basis(U1Irrep)), xxz(U, 1, 2)), 4)
        end,
        "XXZ + field, L=2 (U1)" => function ()
            U = spin_ops(U1Irrep)
            BU1 = spin_basis(U1Irrep)
            return (
                opsum(
                    InfiniteChain([BU1, BU1]), xxz(U, 1, 2; Δ = 0.7), xxz(U, 2, 3; Δ = 0.7), 0.3 * U.S_z[1]
                ), 2,
            )
        end,
        "t-V fermions (U1)" => function ()
            F1 = fermion_ops(U1Irrep)
            return (opsum(InfiniteChain(fermion_basis(U1Irrep)), tV(F1, 1, 2)), 4)
        end,
        "t-V fermions, L=2 (U1)" => function ()
            F1 = fermion_ops(U1Irrep)
            BF = fermion_basis(U1Irrep)
            return (opsum(InfiniteChain([BF, BF]), tV(F1, 1, 2), tV(F1, 2, 3; V = 1.0)), 2)
        end,
        "decaying XXZ (U1)" => function ()
            U = spin_ops(U1Irrep)
            return (opsum(InfiniteChain(spin_basis(U1Irrep)), expterm(xxz(U, 1, 2); decay = 0.5)), 4)
        end,
        "decaying hopping + NN (fermions)" => function ()
            F1 = fermion_ops(U1Irrep)
            return (
                opsum(
                    InfiniteChain(fermion_basis(U1Irrep)), tV(F1, 1, 2),
                    expterm(F1.f_plus_f_min[1, 2] + F1.f_plus_f_min[2, 1]; decay = 0.4)
                ), 4,
            )
        end,
    ]
    return _select_models(builders, names)
end

# Infinite chains: contract `k` unit cells with the level `s` of the bond on the left and the level `f` on the right,
# giving a dense window operator (`secs` are the bond sectors of the last site of the cell).
function select_left(T, sec, k)
    I = eltype(sec)
    sec[k] == unit(I) || error("not a neutral index")
    P = zeros(ComplexF64, Vect[I](unit(I) => 1) ← bond_space(sec))
    block(P, unit(I))[1, deg_indices(sec)[k]] = 1
    @tensor R[a o; i br] := P[a; bl] * T[bl o; i br]
    return R
end
function select_right(T, sec, k)
    I = eltype(sec)
    sec[k] == unit(I) || error("not a neutral index")
    Q = zeros(ComplexF64, bond_space(sec) ← Vect[I](unit(I) => 1))
    block(Q, unit(I))[deg_indices(sec)[k], 1] = 1
    @tensor R[bl o; i a] := T[bl o; i br] * Q[br; a]
    return R
end
function window_operator(Ts, secs, s, f, cells)
    L = length(Ts)
    window = repeat(Ts, cells)
    window[1] = select_left(window[1], secs[L], s)
    window[end] = select_right(window[end], secs[L], f)
    return plain(mpo_tensormap(window))
end

# Terms an infinite MPO generates in a window of `ncells` cells (cell tiled, paths start → finish). A
# sandwich against `window_terms`, not an equality: everything produced is a translate in the window at
# its exact coefficient, but a path may run up to `R` sites past its term's support (the pending/started
# collision), so only translates ending `R + 1` sites before the right edge are all produced.
function mpo_terms_window(H::SymbolicMPO{I, <:InfiniteChain}, ncells::Int) where {I}
    L = length(H)
    N = ncells * L
    Ws = [H.Ws[mod1(j, L)] for j in 1:N]
    secs = [H.bondsectors[mod1(j, L)] for j in 1:N]
    return OpSum.reduced_terms(Ws, secs; leftidx = H.start[L], rightidx = H.finish[L])
end

# a term sum as a dictionary `support => coefficient`; `Term` equality includes the coefficient
termdict(ts) = Dict(OpSum.support(t) => t.coeff for t in ts)
