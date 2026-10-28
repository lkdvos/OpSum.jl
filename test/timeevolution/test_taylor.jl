using Test
using OpSum
using OpSum: mpo, mpo_tensors, opsum, expterm, SiteOperator, storedpairs, TaylorEntry, evaluate, scaled, site_product,
    jordan_layout, close_right, mpo_power, taylor_work, run_passes, fold_passive!, merge_equivalent!, prune_orphans!,
    extend!, compress_levels!, _kron_entry!, rowbond, bond_space, deg_indices
using TensorKit
using TensorKit: @tensor
using LinearAlgebra: norm

include(joinpath(@__DIR__, "..", "testutils.jl"))

# The Taylor MPO (`taylor_mpo`): Alg. 1 and 2 (the structure against the paper's tables, the exactness of the
# merge), Alg. 3 and 4 (the lazy entries of the next power, the closed forms of the paper and of MPSKit's tests),
# and the order of accuracy against the dense exponential.

taylor_dense(T, τ) = plain(mpo_tensormap(mpo_tensors(T, τ)))
const NOCOMPRESS = (fold_passive!,)

# the models by name (see `finite_models` in testutils.jl)
ising() = only(finite_models("Ising (Trivial)"))
const CHEAP = ("Ising (Trivial)",)

@testset "TaylorEntry" begin
    A = jordan_layout(mpo(ising()[2])).Ws[2][1, end]   # a one-letter operator
    z = zero(TaylorEntry{Trivial})
    @test iszero(z) && iszero(TaylorEntry(zero(SiteOperator{Trivial})))
    e = TaylorEntry(A, 1) + TaylorEntry(A * 2.0, 3) + TaylorEntry(one(SiteOperator{Trivial}))
    @test e.powers == [0, 1, 3]
    @test (e + z) == e && (z + e) == e
    # the same power accumulates, a cancelled one disappears
    f = e + TaylorEntry(-A, 1)
    @test f.powers == [0, 3]
    @test iszero(e + scaled(e, -1))
    # monomial scaling shifts the powers
    g = scaled(e, 0.5, 2)
    @test g.powers == [2, 3, 5] && g.ops[2] ≈ A * 0.5
    @test iszero(scaled(e, 0))
    # evaluation
    τ = 0.3 - 0.2im
    @test evaluate(e, τ) ≈ one(SiteOperator{Trivial}) + A * τ + A * (2τ^3)
    @test iszero(evaluate(z, τ))
end

@testset "closed power: the right boundary has both identity channels" begin
    H = ising()[2]
    W = mpo(H)
    L = length(W)
    J = close_right(jordan_layout(W))
    @test J.bondsectors[L + 1] == fill(unit(sectortype(H.lattice)), 2)
    @test J.start[L + 1] == 1 && J.finish[L + 1] == 2
    @test isone(J.Ws[L][1, 1])
    D = densify(H)
    for N in 1:3
        P = mpo_power(W, N; closed = true)
        @test length(P.levels[L]) == 2^N
        Ts = mpo_tensors(P.mpo)
        sec = P.mpo.bondsectors[L]
        # the finish selects the power of H, the start the identity
        for (k, expected) in ((2^N, D^N), (1, one(D)))
            Q = zeros(ComplexF64, bond_space(sec) ← Vect[sectortype(H.lattice)](unit(sectortype(H.lattice)) => 1))
            block(Q, unit(sectortype(H.lattice)))[deg_indices(sec)[k], 1] = 1
            T = copy(Ts)
            @tensor R[bl o; i a] := T[end][bl o; i br] * Q[br; a]
            T[end] = R
            @test plain(mpo_tensormap(T)) ≈ expected
        end
    end
end

# the bond matrix entry of the (not yet compacted) work at site `i`, between levels given as tuples
function level_entry(w, i, row, col)
    rb = rowbond(w, i)
    return get(w.Ms[i].rows[rb == 0 ? 1 : w.index[rb][row]], w.index[i][col], zero(TaylorEntry{eltype(w.secs[1])}))
end

# a polynomial from `power => operator` pairs
poly(I, pairs...) = foldl(+, (TaylorEntry(op, p) for (p, op) in pairs); init = zero(TaylorEntry{I}))

@testset "Alg 1 at N = 2 reproduces Eqs. (30)/(32): $name" for (name, H) in
    finite_models("Ising (Trivial)", "J1-J2 (U1)")
    W = mpo(H)
    I = sectortype(H.lattice)
    i = 3
    B = H.lattice[i]
    J = jordan_layout(W)
    n = length(J.bondsectors[i + 1])
    Wb = J.Ws[i]
    @test size(Wb) == (length(J.bondsectors[i]), n)
    op(l, r) = Wb[l, r]
    mul(x, y) = site_product(B, x, y)
    mids = 2:(n - 1)
    D, Id = op(1, n), op(1, 1)
    w = taylor_work(W, 2)
    @test fold_passive!(w) == 3 * length(W)
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
    finite_models("Ising (Trivial)", "XXZ (U1)", "J1-J2 (U1)")
    W = mpo(H)
    I = sectortype(H.lattice)
    i = 3
    B = H.lattice[i]
    J = jordan_layout(W)
    n = length(J.bondsectors[i + 1])
    chi = n - 2
    op(l, r) = J.Ws[i][l, r]
    mul(x, y) = site_product(B, x, y)
    mids = 2:(n - 1)
    D, Id = op(1, n), op(1, 1)

    # N = 1: [[1 + τD, C], [τB, A]]
    w1 = taylor_work(W, 1)
    fold_passive!(w1)
    @test merge_equivalent!(w1) == 0
    prune_orphans!(w1)
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
    w2 = taylor_work(W, 2)
    @test fold_passive!(w2) == 3 * length(W)
    @test merge_equivalent!(w2) > 0
    prune_orphans!(w2)
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

@testset "bond dimensions" begin
    # Alg. 1 + 2: N = 1: 1 + χ, N = 2: 1 + 2χ + χ² (the paper); N = 3, 4 are the observed counts (Ising). A nilpotent
    # letter (`S⁺S⁺ = 0`, charge conservation) leaves some levels unreachable, which are pruned: below the formula.
    for (name, higher) in (("Ising (Trivial)", [8, 15]), ("XXZ (U1)", nothing), ("J1-J2 (U1)", nothing))
        W = mpo(only(finite_models(name))[2])
        chi = length(jordan_layout(W).bondsectors[4]) - 2
        for N in 1:2
            D = bonddims(taylor_mpo(W, N; extension = false))
            full = N == 1 ? 1 + chi : 1 + 2chi + chi^2
            @test (name == "Ising (Trivial)" ? D[3] == full : D[3] <= full)
            @test D[end] == 1
        end
        higher === nothing && continue
        @test chi == 1
        for (N, d) in zip(3:4, higher)
            @test bonddims(taylor_mpo(W, N; extension = false))[3] == d
        end
    end

    # with the extension and compression (the paper's counts, Sec. 8.3, in the interior Jordan dimension `χ`)
    paper(chi, N) = (
        1 + chi, 1 + chi + chi^2, 1 + 3chi + chi^2 + chi^3, 1 + 5chi + 4chi^2 + chi^3 + chi^4,
    )[N]
    for (name, observed) in (("Ising (Trivial)", [2, 3, 6, 12]), ("XXZ (U1)", [4, 13, 42, 130]))
        W = mpo(only(finite_models(name))[2])
        chi = length(jordan_layout(W).bondsectors[4]) - 2
        name == "Ising (Trivial)" && @test chi == 1
        for N in 1:4
            D = bonddims(taylor_mpo(W, N; extension = true, compression = true))
            @test D[3] == observed[N]
            @test D[3] <= paper(chi, N)
            name == "Ising (Trivial)" && @test D[3] == paper(chi, N)
            @test D[end] == 1
        end
        # the extension adds entries between existing levels, which can only make more of them reachable
        @test bonddims(taylor_mpo(W, 2; extension = false))[3] <=
            bonddims(taylor_mpo(W, 2; extension = true))[3] <= 1 + 2chi + chi^2
    end
end

# Alg. 2 is exact after Alg. 1. After the extension (Alg. 3) only the merge of the start channels is exact; the merge
# of the finish channels changes the terms beyond `τ^N`, exactly the ones the extension adds partially (see `taylor.jl`)
# (Ising at N = 2, 3; the unmerged `H^3` on five sites is the expensive dense oracle, so the others stay at N = 2)
@testset "Alg 2 is exact: $name" for (name, H) in finite_models("Ising (Trivial)", "t-V fermions (U1)")
    W = mpo(H)
    D = densify(H)
    merge_starts!(w) = merge_equivalent!(w; right = false)
    taus = (-0.2im, 0.1 + 0.15im)
    for N in (name == "Ising (Trivial)" ? (2, 3) : (2,))
        T1 = run_passes(W, N, NOCOMPRESS)
        T12 = taylor_mpo(W, N; extension = false)
        @test all(bonddims(T12)[2:(end - 1)] .< bonddims(T1)[2:(end - 1)])
        for τ in taus
            @test taylor_dense(T12, τ) ≈ taylor_dense(T1, τ)
        end

        E1 = run_passes(W, N, (extend!, fold_passive!))
        Es = run_passes(W, N, (extend!, fold_passive!, merge_starts!))
        Ef = run_passes(W, N, (extend!, fold_passive!, merge_equivalent!))
        @test all(bonddims(Es)[2:(end - 1)] .< bonddims(E1)[2:(end - 1)])
        @test all(bonddims(Ef)[2:(end - 1)] .<= bonddims(Es)[2:(end - 1)])
        for τ in taus
            @test taylor_dense(Es, τ) ≈ taylor_dense(E1, τ)
        end
        # the full merge agrees through `τ^N`: the difference is at least of order `τ^{N+1}`
        cs = (0.4, 0.2, 0.1)
        diffs = [norm(taylor_dense(Ef, -c / norm(D)) - taylor_dense(E1, -c / norm(D))) for c in cs]
        if all(>(1.0e-13), diffs)
            @test (log(diffs[3]) - log(diffs[1])) / (log(cs[3]) - log(cs[1])) >= N + 1 - 0.15
        end
    end
end

@testset "lazy entries of the next power: $name" for (name, H) in
    finite_models(
        "Ising (Trivial)", "XXZ (U1)", "t-V fermions (U1)", "t-V fermions (parity), next-nearest hop",
        "Hubbard (U1 x U1)",
    )
    W = mpo(H)
    L = length(W)
    # the tries of the second power evaluate the entries of the third (words are order-free), signs included
    P2 = mpo_power(W, 2; closed = true)
    P3 = mpo_power(W, 3; closed = true)
    ncompared = 0
    for i in (1, 2, L)
        rows = i == 1 ? [(1, 1, 1)] : P3.levels[i - 1]
        cols = P3.levels[i]
        rowindex = Dict(t => k for (k, t) in enumerate(rows))
        colindex = Dict(t => k for (k, t) in enumerate(cols))
        stored = Dict(Tuple(idx) => op for (idx, op) in storedpairs(P3.mpo.Ws[i]))
        for l in rows, r in cols
            x = _kron_entry!(P2.sites[i], l, r)
            y = get(stored, (rowindex[l], colindex[r]), nothing)
            if y === nothing
                @test x === nothing
            else
                @test x ≈ y
                ncompared += 1
            end
        end
    end
    @test ncompared > 0
    # a layer of identities is the identity
    @test _kron_entry!(P2.sites[2], (1,), (1,)) ≈ one(SiteOperator{sectortype(H.lattice)})
end

@testset "Alg 3 reads a fraction of the next power: $name" for (name, H) in finite_models("Ising (Trivial)", "XXZ (U1)")
    W = mpo(H)
    for N in 1:3
        w = taylor_work(W, N)
        nmul = w.stats.nmul
        nread = extend!(w)
        full = mpo_power(W, N + 1; closed = true).stats
        @test 0 < nread < full.nentries
        # the products are looked up in the tries of the power, which the next power would build anew
        @test w.stats.nmul - nmul <= full.nmul
    end
end

# the mean of the on-site products of the given words
function mean_words(B, words...)
    return foldl(+, (foldl((x, y) -> site_product(B, x, y), wd) for wd in words)) * (1 / length(words))
end

@testset "Alg 3 and 4 closed forms (Eqs. (43), (45), MPSKit's `O(1)`/`O(2) exact expression`): $name" for (name, H) in
    finite_models("Ising (Trivial)", "XXZ (U1)", "J1-J2 (U1)")
    W = mpo(H)
    I = sectortype(H.lattice)
    i = 3
    B = H.lattice[i]
    J = jordan_layout(W)
    n = length(J.bondsectors[i + 1])
    mids = 2:(n - 1)
    D, Id = J.Ws[i][1, n], J.Ws[i][1, 1]
    mul(x, y) = site_product(B, x, y)
    avg(words...) = mean_words(B, words...)
    Bv(a) = J.Ws[i][a, n]
    Cv(b) = J.Ws[i][1, b]
    Av(a, b) = J.Ws[i][a, b]

    function passes(N)
        w = taylor_work(W, N)
        @test extend!(w) > 0
        fold_passive!(w)
        merge_equivalent!(w)
        return w
    end

    # N = 1: the extension alone (there is nothing to compress)
    w1 = passes(1)
    @test compress_levels!(w1) == 0
    prune_orphans!(w1)
    @test level_entry(w1, i, (1,), (1,)) ≈ poly(I, 0 => Id, 1 => D, 2 => mul(D, D) * 0.5)
    for a in mids
        @test level_entry(w1, i, (1,), (a,)) ≈ poly(I, 0 => Cv(a), 1 => avg((Cv(a), D), (D, Cv(a))))
        @test level_entry(w1, i, (a,), (1,)) ≈ poly(I, 1 => Bv(a), 2 => avg((Bv(a), D), (D, Bv(a))))
        for b in mids
            sym_AD = avg((Av(a, b), D), (D, Av(a, b)))
            sym_CB = avg((Cv(b), Bv(a)), (Bv(a), Cv(b)))
            @test level_entry(w1, i, (a,), (b,)) ≈ poly(I, 0 => Av(a, b), 1 => sym_AD + sym_CB)
        end
    end

    # N = 2 with the compression: the levels (1,1), (1,c), (c,d); the sums over the orderings of the paper, each
    # letter carrying the index of its position among the letters that have one
    w2 = passes(2)
    @test compress_levels!(w2) > 0
    prune_orphans!(w2)
    alive = Set(t for (t, a) in zip(w2.levels[i], w2.alive[i]) if a)
    canonical = Set([[(1, 1)]; [(1, a) for a in mids]; [(a, b) for a in mids for b in mids]])
    @test alive ⊆ canonical
    name == "Ising (Trivial)" && @test alive == canonical && count(w2.alive[i]) == 1 + 1 * (n - 2) + (n - 2)^2
    # the rows are the levels of the bond to the left, which can lose other levels to the pruning
    live(t, bond) = w2.alive[bond][w2.index[bond][t]]
    function chk(row, col, expected)
        (live(row, i - 1) && live(col, i)) || return nothing
        @test level_entry(w2, i, row, col) ≈ expected
        return nothing
    end
    chk(
        (1, 1), (1, 1),
        poly(I, 0 => Id, 1 => D, 2 => mul(D, D) * 0.5, 3 => mul(mul(D, D), D) * (1 / 6))
    )
    for b in mids
        chk(
            (1, 1), (1, b),
            poly(
                I, 0 => Cv(b), 1 => avg((Cv(b), D), (D, Cv(b))),
                2 => avg((Cv(b), D, D), (D, Cv(b), D), (D, D, Cv(b))) * 0.5
            )
        )
    end
    for c in mids
        chk(
            (1, c), (1, 1),
            poly(
                I, 1 => Bv(c), 2 => avg((Bv(c), D), (D, Bv(c))),
                3 => avg((Bv(c), D, D), (D, Bv(c), D), (D, D, Bv(c))) * 0.5
            )
        )
    end
    for a in mids, b in mids
        chk(
            (1, 1), (a, b),
            poly(I, 0 => mul(Cv(a), Cv(b)), 1 => avg((Cv(a), Cv(b), D), (Cv(a), D, Cv(b)), (D, Cv(a), Cv(b))))
        )
    end
    for c in mids, d in mids
        chk(
            (c, d), (1, 1),
            poly(
                I, 2 => mul(Bv(c), Bv(d)) * 0.5,
                3 => avg((Bv(c), Bv(d), D), (Bv(c), D, Bv(d)), (D, Bv(c), Bv(d))) * 0.5
            )
        )
    end
    for c in mids, b in mids
        A, Bc, Cb = Av(c, b), Bv(c), Cv(b)
        chk(
            (1, c), (1, b),
            poly(
                I, 0 => A,
                1 => avg((A, D), (D, A)) + avg((Cb, Bc), (Bc, Cb)),
                2 => avg((A, D, D), (D, A, D), (D, D, A)) * 0.5 +
                    avg((Cb, Bc, D), (Cb, D, Bc), (Bc, Cb, D), (Bc, D, Cb), (D, Cb, Bc), (D, Bc, Cb))
            )
        )
        for a in mids
            chk(
                (1, c), (a, b),
                poly(
                    I,
                    0 => avg((Cv(a), Av(c, b)), (Av(c, a), Cb)) * 2,
                    1 => avg(
                        (Av(c, a), Cb, D), (Av(c, a), D, Cb), (Cv(a), Av(c, b), D),
                        (Cv(a), D, Av(c, b)), (D, Av(c, a), Cb), (D, Cv(a), Av(c, b))
                    ) * 2 + avg((Cv(a), Bc, Cb), (Cv(a), Cb, Bc), (Bc, Cv(a), Cb))
                )
            )
        end
    end
    for c in mids, d in mids, b in mids
        chk(
            (c, d), (1, b),
            poly(
                I,
                1 => avg((Av(c, b), Bv(d)), (Bv(c), Av(d, b))),
                2 => avg(
                    (Av(c, b), Bv(d), D), (Av(c, b), D, Bv(d)), (Bv(c), Av(d, b), D),
                    (Bv(c), D, Av(d, b)), (D, Av(c, b), Bv(d)), (D, Bv(c), Av(d, b))
                ) + avg((Bv(c), Bv(d), Cv(b)), (Bv(c), Cv(b), Bv(d)), (Cv(b), Bv(c), Bv(d))) * 0.5
            )
        )
        for a in mids
            chk(
                (c, d), (a, b),
                poly(
                    I,
                    0 => mul(Av(c, a), Av(d, b)),
                    1 => avg((Av(c, a), Av(d, b), D), (Av(c, a), D, Av(d, b)), (D, Av(c, a), Av(d, b))) +
                        avg(
                        (Av(c, a), Cv(b), Bv(d)), (Av(c, a), Bv(d), Cv(b)), (Cv(a), Av(c, b), Bv(d)),
                        (Cv(a), Bv(c), Av(d, b)), (Bv(c), Av(d, a), Cv(b)), (Bv(c), Cv(a), Av(d, b))
                    ) * 2
                )
            )
        end
    end
end

# `E(τ) = ‖U(τ) − exp(τH)‖` (`U(τ)` the dense operator `U(τ)`) at `τ = -c / ‖H‖`, imaginary or real time
function error_at(U, D, c, imaginary = true)
    τ = imaginary ? -c / norm(D) : -im * c / norm(D)
    return norm(U(τ) - exp(τ * D))
end

# the slope of the log-log fit of `E(τ)` against `τ ∈ c / ‖H‖`
function error_slope(U, D, imaginary::Bool)
    cs = (0.4, 0.2, 0.1)
    errs = [error_at(U, D, c, imaginary) for c in cs]
    @test all(>(1.0e-13), errs)
    xs = log.(cs)
    ys = log.(errs)
    return sum((xs .- sum(xs) / 3) .* (ys .- sum(ys) / 3)) / sum((xs .- sum(xs) / 3) .^ 2)
end

# order of accuracy N + 1 at N = 2 with all toggles, a bosonic and a fermionic model (N = 1, 3, 4, real time and
# the extension's precision on the cheap ones; the operator itself is pinned to MPSKit's `TaylorCluster` in
# test_mpskit_ext.jl)
@testset "order of accuracy: $name" for (name, H) in finite_models("Ising (Trivial)", "t-V fermions (U1)")
    D = densify(H)
    W = mpo(H)
    for extension in (false, true), compression in (false, true)
        T = taylor_mpo(W, 2; extension, compression)
        @test error_slope(τ -> taylor_dense(T, τ), D, true) ≈ 3 atol = 0.15
    end
    # the defaults are the paper's extended MPO
    @test taylor_dense(taylor_mpo(W, 2), 0.1) ≈ taylor_dense(taylor_mpo(W, 2; extension = true, compression = false), 0.1)
    if name in CHEAP
        for N in 1:4, (extension, compression) in ((false, false), (true, true))
            T = taylor_mpo(W, N; extension, compression)
            U(τ) = taylor_dense(T, τ)
            @test error_slope(U, D, true) ≈ N + 1 atol = 0.15
            N <= 3 && @test error_slope(U, D, false) ≈ N + 1 atol = 0.15
        end
        # the extension is more precise than the plain truncation
        for N in 1:3
            Up = taylor_mpo(W, N; extension = false)
            Ue = taylor_mpo(W, N; extension = true)
            @test error_at(τ -> taylor_dense(Ue, τ), D, 0.4) < error_at(τ -> taylor_dense(Up, τ), D, 0.4)
        end
    end
end

@testset "taylor_mpo(H, N) and the call syntax" begin
    H = ising()[2]
    T = taylor_mpo(H, 2)
    @test T isa TaylorMPO
    @test T.order == 2
    @test taylor_dense(T, 0.1) ≈ taylor_dense(taylor_mpo(mpo(H), 2), 0.1)
    @test T(0.1) == evaluate(T, 0.1)
    @test length(T(0.1)) == length(T)
    @test T.bondsectors[end] == [unit(Trivial)]
    @test size(T.Ws[1], 1) == 1 && size(T.Ws[end], 2) == 1
    # τ = 0 is the identity
    @test taylor_dense(T, 0.0) ≈ one(densify(H))
    @test occursin("order = 2", sprint(show, T))
end

# a decaying channel on a finite chain has a Jordan form, so it has a Taylor MPO of the right order
@testset "finite chain with a decaying channel" begin
    S = spin_ops(U1Irrep)
    B = spin_basis(U1Irrep)
    h = 0.5 * (S.S_plus_S_min[1, 2] + S.S_min_S_plus[1, 2]) + S.S_z_S_z[1, 2]
    Hd = opsum(FiniteChain(B, 4), expterm(h; decay = 0.5))
    D = densify(Hd)
    for N in 1:2
        T = taylor_mpo(Hd, N)
        @test error_slope(τ -> taylor_dense(T, τ), D, true) ≈ N + 1 atol = 0.15
    end
end

@testset "errors" begin
    S = spin_ops(U1Irrep)
    B = spin_basis(U1Irrep)
    H = opsum(
        FiniteChain(B, 4),
        (0.5 * (S.S_plus_S_min[i, i + 1] + S.S_min_S_plus[i, i + 1]) + S.S_z_S_z[i, i + 1] for i in 1:3)
    )
    @test_throws ArgumentError taylor_mpo(H, 0)
    SU = spin_ops(SU2Irrep)
    Hs = opsum(FiniteChain(spin_basis(SU2Irrep), 3), (SU.S_exchange[i, i + 1] for i in 1:2))
    @test_throws ArgumentError taylor_mpo(Hs, 2)
end

# Infinite chains: a window of `cells` unit cells with the start level `1` on the left and the folded level `1` on
# the right (for the Hamiltonian itself the start and finish channels).
@testset "infinite chain window: $name" for (name, (H, cells)) in infinite_models(
        "Heisenberg (U1)", "XXZ + field, L=2 (U1)", "t-V fermions (U1)", "t-V fermions, L=2 (U1)", "decaying XXZ (U1)",
    )
    W = mpo(H)
    L = length(W)
    Hwin = window_operator(mpo_tensors(W), W.bondsectors, W.start[L], W.finish[L], cells)
    # (the extension adds terms of order N + 2 whose weight can pull the fit below N + 1, never above)
    for N in 1:3, (extension, compression) in ((false, false), (true, true))
        T = taylor_mpo(W, N; extension, compression)
        @test T.bondsectors[L][1] == unit(sectortype(H.lattice))
        Tw(τ) = window_operator(mpo_tensors(T, τ), T.bondsectors, 1, 1, cells)
        @test Tw(0.0) ≈ one(Hwin)
        @test N + 1 - 0.3 <= error_slope(Tw, Hwin, true) <= N + 1 + 0.15
        if N == 2 && extension
            @test N + 1 - 0.3 <= error_slope(Tw, Hwin, false) <= N + 1 + 0.15
        end
    end
end
