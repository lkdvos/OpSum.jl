using Test
using OpSum
using OpSum: mpo, irrep_mpo_tensors, instantiate, couple, opsum, expterm, SiteOperator, storedpairs, prune,
    TaylorEntry, evaluate, _scaled, _mul, _jordan_layout, _mpo_power, _taylor_work, _taylor_mpo, _fold_passive!,
    _merge_equivalent!, _prune_orphans!, _rowbond, _bond_space, _deg_indices
using TensorKit
using TensorKit: @tensor
using LinearAlgebra: norm

include(joinpath(@__DIR__, "testutils.jl"))   # plain

# Alg. 1 and 2 of the Taylor MPO (`taylor_mpo`): the structure against the paper's tables, the exactness of the
# merge, and the order of accuracy against the dense exponential.

dense(H) = plain(instantiate(H))
taylor_dense(T, τ) = plain(mpo_tensormap(irrep_mpo_tensors(T, τ)))
const NOCOMPRESS = (_fold_passive!,)

function finite_models()
    S = spin_ops(Trivial)
    U = spin_ops(U1Irrep)
    F1 = fermion_ops(U1Irrep)
    F0 = fermion_ops()
    Hb = hubbard_ops(U1Irrep, U1Irrep)
    chain(B, N) = FiniteChain(B, N)
    xxz(i, j) = 0.5 * (U.S_plus_S_min[i, j] + U.S_min_S_plus[i, j]) + 0.8 * U.S_z_S_z[i, j]
    return [
        "Ising (Trivial)" => opsum(
            chain(spin_basis(Trivial), 5),
            (-S.S_z_S_z[i, i + 1] for i in 1:4), (0.7 * S.S_x[i] for i in 1:5)
        ),
        "XXZ (U1)" => opsum(
            chain(spin_basis(U1Irrep), 5), (xxz(i, i + 1) for i in 1:4), (0.3 * U.S_z[i] for i in 1:5)
        ),
        "J1-J2 (U1)" => opsum(
            chain(spin_basis(U1Irrep), 5), (xxz(i, i + 1) for i in 1:4), (0.4 * xxz(i, i + 2) for i in 1:3)
        ),
        "t-V fermions (U1)" => opsum(
            chain(fermion_basis(U1Irrep), 5),
            (F1.f_plus_f_min[i, i + 1] + F1.f_plus_f_min[i + 1, i] + 2 * F1.f_num[i] * F1.f_num[i + 1] for i in 1:4)
        ),
        "t-V fermions (parity), next-nearest hop" => opsum(
            chain(fermion_basis(), 4),
            (F0.f_plus_f_min[i, i + 1] + F0.f_plus_f_min[i + 1, i] + 2 * F0.f_num[i] * F0.f_num[i + 1] for i in 1:3),
            (0.3 * (F0.f_plus_f_min[i, i + 2] + F0.f_plus_f_min[i + 2, i]) for i in 1:2)
        ),
        "Hubbard (U1 x U1)" => opsum(
            chain(hubbard_basis(U1Irrep, U1Irrep), 3),
            (
                Hb.u_plus_u_min[i, i + 1] + Hb.u_min_u_plus[i, i + 1] +
                    Hb.d_plus_d_min[i, i + 1] + Hb.d_min_d_plus[i, i + 1] for i in 1:2
            ),
            (4.0 * Hb.ud_num[i] for i in 1:3)
        ),
    ]
end

@testset "TaylorEntry" begin
    A = _jordan_layout(mpo(finite_models()[1][2])).Ws[2][1, end]   # a one-letter operator
    z = zero(TaylorEntry{Trivial})
    @test iszero(z) && iszero(TaylorEntry(zero(SiteOperator{Trivial})))
    e = TaylorEntry(A, 1) + TaylorEntry(A * 2.0, 3) + TaylorEntry(one(SiteOperator{Trivial}))
    @test e.powers == [0, 1, 3]
    @test (e + z) == e && (z + e) == e
    # the same power accumulates, a cancelled one disappears
    f = e + TaylorEntry(-A, 1)
    @test f.powers == [0, 3]
    @test iszero(e + _scaled(e, -1))
    # monomial scaling shifts the powers
    g = _scaled(e, 0.5, 2)
    @test g.powers == [2, 3, 5] && g.ops[2] ≈ A * 0.5
    @test iszero(_scaled(e, 0))
    # evaluation
    τ = 0.3 - 0.2im
    @test evaluate(e, τ) ≈ one(SiteOperator{Trivial}) + A * τ + A * (2τ^3)
    @test iszero(evaluate(z, τ))
end

@testset "closed power: the right boundary has both identity channels" begin
    H = Dict(finite_models())["t-V fermions (U1)"]
    W = mpo(H)
    L = length(W)
    J = _jordan_layout(W; closed = true)
    @test J.bondsectors[L + 1] == fill(unit(sectortype(H.lattice)), 2)
    @test J.start[L + 1] == 1 && J.finish[L + 1] == 2
    @test isone(J.Ws[L][1, 1])
    D = dense(H)
    for N in 1:3
        P = _mpo_power(W, N; closed = true)
        @test length(P.levels[L]) == 2^N
        Ts = irrep_mpo_tensors(P.mpo)
        sec = P.mpo.bondsectors[L]
        # the finish selects the power of H, the start the identity
        for (k, expected) in ((2^N, D^N), (1, one(D)))
            Q = zeros(ComplexF64, _bond_space(sec) ← Vect[sectortype(H.lattice)](unit(sectortype(H.lattice)) => 1))
            block(Q, unit(sectortype(H.lattice)))[_deg_indices(sec)[k], 1] = 1
            T = copy(Ts)
            @tensor R[bl o; i a] := T[end][bl o; i br] * Q[br; a]
            T[end] = R
            @test plain(mpo_tensormap(T)) ≈ expected
        end
    end
end

# the bond matrix entry of the (not yet compacted) work at site `i`, between levels given as tuples
function level_entry(w, i, row, col)
    rb = _rowbond(w, i)
    return get(w.Ms[i].rows[rb == 0 ? 1 : w.index[rb][row]], w.index[i][col], zero(TaylorEntry{eltype(w.secs[1])}))
end

# a polynomial from `power => operator` pairs
poly(I, pairs...) = foldl(+, (TaylorEntry(op, p) for (p, op) in pairs); init = zero(TaylorEntry{I}))

@testset "Alg 1 at N = 2 reproduces Eqs. (30)/(32): $name" for (name, H) in
    (finite_models()[1], finite_models()[3])
    W = mpo(H)
    I = sectortype(H.lattice)
    i = 3
    B = H.lattice[i]
    J = _jordan_layout(W)
    n = length(J.bondsectors[i + 1])
    Wb = J.Ws[i]
    @test size(Wb) == (length(J.bondsectors[i]), n)
    op(l, r) = Wb[l, r]
    mul(x, y) = _mul(B, x, y)
    mids = 2:(n - 1)
    D, Id = op(1, n), op(1, 1)
    w = _taylor_work(W, 2)
    @test _fold_passive!(w) == 3 * length(W)
    # (32): the first column collects the folded columns with the weights 1, τ/2, τ/2, τ²/2
    @test level_entry(w, i, (1, 1), (1, 1)) ≈ poly(I, 0 => Id, 1 => D, 2 => mul(D, D) * 0.5)
    for a in mids
        Ba = op(a, n)
        @test level_entry(w, i, (1, a), (1, 1)) ≈ poly(I, 1 => Ba * 0.5, 2 => mul(D, Ba) * 0.5)
        @test level_entry(w, i, (a, 1), (1, 1)) ≈ poly(I, 1 => Ba * 0.5, 2 => mul(Ba, D) * 0.5)
        @test level_entry(w, i, (a, n), (1, 1)) ≈ poly(I, 2 => Ba * 0.5)
        @test level_entry(w, i, (n, a), (1, 1)) ≈ poly(I, 2 => Ba * 0.5)
        for b in mids
            @test level_entry(w, i, (a, b), (1, 1)) ≈ poly(I, 2 => mul(Ba, op(b, n)) * 0.5)
        end
    end
    # the remaining columns are those of H², untouched (Eq. (22) with (1,3), (3,1), (3,3) removed)
    for a in mids, b in mids
        @test level_entry(w, i, (1, 1), (a, b)) ≈ poly(I, 0 => mul(op(1, a), op(1, b)))
        @test level_entry(w, i, (a, 1), (a, b)) ≈ poly(I, 0 => mul(op(a, a), op(1, b)))
    end
    # the folded levels are gone
    @test count(w.alive[i]) == n^2 - 3
    @test all(!w.alive[i][w.index[i][t]] for t in ((1, n), (n, 1), (n, n)))
end

@testset "Alg 1 + 2 at N = 1 and N = 2 reproduce Eqs. (27) and (39): $name" for (name, H) in
    (finite_models()[1], finite_models()[2], finite_models()[3])
    W = mpo(H)
    I = sectortype(H.lattice)
    i = 3
    B = H.lattice[i]
    J = _jordan_layout(W)
    n = length(J.bondsectors[i + 1])
    chi = n - 2
    op(l, r) = J.Ws[i][l, r]
    mul(x, y) = _mul(B, x, y)
    mids = 2:(n - 1)
    D, Id = op(1, n), op(1, 1)

    # N = 1: [[1 + τD, C], [τB, A]]
    w1 = _taylor_work(W, 1)
    _fold_passive!(w1)
    @test _merge_equivalent!(w1) == 0
    _prune_orphans!(w1)
    @test count(w1.alive[i]) == 1 + chi
    @test level_entry(w1, i, (1,), (1,)) ≈ poly(I, 0 => Id, 1 => D)
    for a in mids
        @test level_entry(w1, i, (1,), (a,)) ≈ poly(I, 0 => op(1, a))
        @test level_entry(w1, i, (a,), (1,)) ≈ poly(I, 1 => op(a, n))
        for b in mids
            @test level_entry(w1, i, (a,), (b,)) ≈ poly(I, 0 => op(a, b))
        end
    end

    # N = 2: the levels (1,1), (1,a), (a,b), (a,n) of Eq. (39). A nilpotent letter (`S⁺S⁺ = 0`) leaves some
    # levels unreachable, which are pruned: only the entries between surviving levels are compared.
    w2 = _taylor_work(W, 2)
    @test _fold_passive!(w2) == 3 * length(W)
    @test _merge_equivalent!(w2) > 0
    _prune_orphans!(w2)
    canonical = Set([[(1, 1)]; [(1, a) for a in mids]; [(a, b) for a in mids for b in mids]; [(a, n) for a in mids]])
    alive = Set(t for (t, a) in zip(w2.levels[i], w2.alive[i]) if a)
    @test alive ⊆ canonical
    name == "Ising (Trivial)" && @test alive == canonical && count(w2.alive[i]) == 1 + 2chi + chi^2
    live(t) = w2.alive[i][w2.index[i][t]]
    function chk(row, col, expected)
        (live(row) && live(col)) || return nothing
        @test level_entry(w2, i, row, col) ≈ expected
        return nothing
    end
    chk((1, 1), (1, 1), poly(I, 0 => Id, 1 => D, 2 => mul(D, D) * 0.5))
    for a in mids
        Ba = op(a, n)
        chk((1, a), (1, 1), poly(I, 1 => Ba, 2 => (mul(D, Ba) + mul(Ba, D)) * 0.5))
        chk((a, n), (1, 1), poly(I, 2 => Ba * 0.5))
        chk((1, 1), (1, a), poly(I, 0 => op(1, a)))
        # the columns (a,n) and (n,a) are merged: the finish channel of one layer is followed by the other's letter
        chk((1, 1), (a, n), poly(I, 0 => mul(op(1, a), D) + mul(D, op(1, a))))
        for b in mids
            chk((a, b), (1, 1), poly(I, 2 => mul(op(a, n), op(b, n)) * 0.5))
            chk((1, a), (1, b), poly(I, 0 => op(a, b)))
            chk((a, n), (b, n), poly(I, 0 => op(a, b)))
            chk((1, 1), (a, b), poly(I, 0 => mul(op(1, a), op(1, b))))
            # the merged rows of (1,c) and (c,1): CA + AC, and with the merged columns (a,n), (n,a): CB + AD + DA + BC
            for c in mids
                chk((1, c), (a, b), poly(I, 0 => mul(op(1, a), op(c, b)) + mul(op(c, a), op(1, b))))
                chk(
                    (1, c), (a, n),
                    poly(I, 0 => mul(op(1, a), op(c, n)) + mul(op(c, a), D) + mul(D, op(c, a)) + mul(op(c, n), op(1, a)))
                )
                chk((a, c), (b, n), poly(I, 0 => mul(op(a, b), op(c, n)) + mul(op(a, n), op(c, b))))
                for d in mids
                    chk((a, c), (b, d), poly(I, 0 => mul(op(a, b), op(c, d))))
                end
            end
        end
    end
end

bonddims(T) = map(s -> sum(dim, s), T.bondsectors)

@testset "bond dimensions" begin
    # N = 1: 1 + χ, N = 2: 1 + 2χ + χ² (the paper); N = 3, 4 are the observed counts of Alg. 1 + 2 (Ising)
    for (name, H, higher) in (
            ("Ising", finite_models()[1][2], [8, 15]), ("XXZ", finite_models()[2][2], nothing),
            ("J1-J2", finite_models()[3][2], nothing),
        )
        W = mpo(H)
        chi = length(_jordan_layout(W).bondsectors[4]) - 2
        for N in 1:2
            D = bonddims(taylor_mpo(W, N))
            # (a nilpotent letter, `S⁺S⁺ = 0`, leaves some of the levels unreachable)
            full = N == 1 ? 1 + chi : 1 + 2chi + chi^2
            @test (name == "Ising" ? D[3] == full : D[3] <= full)
            @test D[end] == 1
        end
        higher === nothing && continue
        @test chi == 1
        for (N, d) in zip(3:4, higher)
            @test bonddims(taylor_mpo(W, N))[3] == d
        end
    end
end

@testset "Alg 2 is exact: $name" for (name, H) in finite_models()
    W = mpo(H)
    for N in 2:3
        T1 = _taylor_mpo(W, N, NOCOMPRESS)
        T12 = taylor_mpo(W, N)
        @test all(bonddims(T12) .<= bonddims(T1))
        @test all(bonddims(T12)[2:(end - 1)] .< bonddims(T1)[2:(end - 1)])
        for τ in (0.3, -0.2im, 0.1 + 0.15im)
            @test taylor_dense(T12, τ) ≈ taylor_dense(T1, τ)
        end
    end
end

# `E(τ) = ‖U_N(τ) − exp(τH)‖` against `τ ∈ c / ‖H‖`; the slope of its log-log fit
function error_slope(T, D, imaginary::Bool)
    h = norm(D)
    cs = (0.4, 0.2, 0.1)
    errs = map(cs) do c
        τ = imaginary ? -c / h : -im * c / h
        return norm(taylor_dense(T, τ) - exp(τ * D))
    end
    @test all(>(1.0e-13), errs)
    xs = log.(cs)
    ys = log.(errs)
    return sum((xs .- sum(xs) / 3) .* (ys .- sum(ys) / 3)) / sum((xs .- sum(xs) / 3) .^ 2)
end

@testset "order of accuracy: $name" for (name, H) in finite_models()
    D = dense(H)
    W = mpo(H)
    for N in 1:4
        T = taylor_mpo(W, N)
        for imaginary in (true, false)
            @test error_slope(T, D, imaginary) ≈ N + 1 atol = 0.15
        end
    end
end

@testset "taylor_mpo(H, N) and the call syntax" begin
    H = finite_models()[1][2]
    T = taylor_mpo(H, 2)
    @test T isa TaylorMPO
    @test T.order == 2
    @test taylor_dense(T, 0.1) ≈ taylor_dense(taylor_mpo(mpo(H), 2), 0.1)
    @test T(0.1) == evaluate(T, 0.1)
    @test length(T(0.1)) == length(T)
    @test T.bondsectors[end] == [unit(Trivial)]
    @test size(T.Ws[1], 1) == 1 && size(T.Ws[end], 2) == 1
    # τ = 0 is the identity
    @test taylor_dense(T, 0.0) ≈ one(dense(H))
    @test occursin("order = 2", sprint(show, T))
end

@testset "errors" begin
    S = spin_ops(U1Irrep)
    B = spin_basis(U1Irrep)
    h = 0.5 * (S.S_plus_S_min[1, 2] + S.S_min_S_plus[1, 2]) + S.S_z_S_z[1, 2]
    H = opsum(
        FiniteChain(B, 4),
        (0.5 * (S.S_plus_S_min[i, i + 1] + S.S_min_S_plus[i, i + 1]) + S.S_z_S_z[i, i + 1] for i in 1:3)
    )
    @test_throws ArgumentError taylor_mpo(H, 0)
    Hd = opsum(FiniteChain(B, 4), expterm(h; decay = 0.5))
    @test_throws ArgumentError taylor_mpo(Hd, 2)
    SU = spin_ops(SU2Irrep)
    Hs = opsum(FiniteChain(spin_basis(SU2Irrep), 3), (SU.S_exchange[i, i + 1] for i in 1:2))
    @test_throws ArgumentError taylor_mpo(Hs, 2)
end

# Infinite chains: contract `cells` unit cells with the start level `1` on the left and the folded level `1` on
# the right (for the Hamiltonian itself the start and finish channels).
function select_left(T, sec, k)
    I = eltype(sec)
    sec[k] == unit(I) || error("not a neutral index")
    P = zeros(ComplexF64, Vect[I](unit(I) => 1) ← _bond_space(sec))
    block(P, unit(I))[1, _deg_indices(sec)[k]] = 1
    @tensor R[a o; i br] := P[a; bl] * T[bl o; i br]
    return R
end
function select_right(T, sec, k)
    I = eltype(sec)
    sec[k] == unit(I) || error("not a neutral index")
    Q = zeros(ComplexF64, _bond_space(sec) ← Vect[I](unit(I) => 1))
    block(Q, unit(I))[_deg_indices(sec)[k], 1] = 1
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

function infinite_models()
    U = spin_ops(U1Irrep)
    F1 = fermion_ops(U1Irrep)
    BU1 = spin_basis(U1Irrep)
    BF = fermion_basis(U1Irrep)
    xxz(i, j; Δ = 1.0) = 0.5 * (U.S_plus_S_min[i, j] + U.S_min_S_plus[i, j]) + Δ * U.S_z_S_z[i, j]
    tV(i, j; V = 2.0) = F1.f_plus_f_min[i, j] + F1.f_plus_f_min[j, i] + V * F1.f_num[i] * F1.f_num[j]
    return [
        "Heisenberg (U1)" => (opsum(InfiniteChain(BU1), xxz(1, 2)), 4),
        "XXZ + field, L=2 (U1)" => (
            opsum(InfiniteChain([BU1, BU1]), xxz(1, 2; Δ = 0.7), xxz(2, 3; Δ = 0.7), 0.3 * U.S_z[1]), 2,
        ),
        "t-V fermions (U1)" => (opsum(InfiniteChain(BF), tV(1, 2)), 4),
        "t-V fermions, L=2 (U1)" => (opsum(InfiniteChain([BF, BF]), tV(1, 2), tV(2, 3; V = 1.0)), 2),
        "decaying XXZ (U1)" => (opsum(InfiniteChain(BU1), expterm(xxz(1, 2); decay = 0.5)), 4),
    ]
end

function window_slope(Tw, D, imaginary::Bool)
    h = norm(D)
    cs = (0.4, 0.2, 0.1)
    errs = map(cs) do c
        τ = imaginary ? -c / h : -im * c / h
        return norm(Tw(τ) - exp(τ * D))
    end
    @test all(>(1.0e-13), errs)
    xs = log.(cs)
    ys = log.(errs)
    return sum((xs .- sum(xs) / 3) .* (ys .- sum(ys) / 3)) / sum((xs .- sum(xs) / 3) .^ 2)
end

@testset "infinite chain window: $name" for (name, (H, cells)) in infinite_models()
    W = mpo(H)
    L = length(W)
    Hwin = window_operator(irrep_mpo_tensors(W), W.bondsectors, W.start[L], W.finish[L], cells)
    for N in 1:3
        T = taylor_mpo(W, N)
        @test T.bondsectors[L][1] == unit(sectortype(H.lattice))
        Tw(τ) = window_operator(irrep_mpo_tensors(T, τ), T.bondsectors, 1, 1, cells)
        @test Tw(0.0) ≈ one(Hwin)
        for imaginary in (true, false)
            @test window_slope(Tw, Hwin, imaginary) ≈ N + 1 atol = 0.15
        end
    end
end
