using Test
using OpSum
using OpSum: OperatorSum, opsum!, opsum, irrep_mpo, irrep_mpo_tensors, jordan_mpo_tensors,
    instantiate, islossless, mpo_tensormap, mpo_terms, chain_terms, expterm, ExpSum, Terms, Term,
    FiniteChain, InfiniteChain, FiniteMPO, InfiniteMPO, couple, couple_channels, spin, spin_ops,
    fermion_ops, unitcell_terms
using TensorKit
using LinearAlgebra: dot

include(joinpath(@__DIR__, "testutils.jl"))   # physmatrix, densedim

# `OperatorSum` is the one container behind every `irrep_mpo`: a lattice, a term bag, and the
# exponentially decaying channels. These tests pin the four cells of the finite/infinite × plain/
# decaying grid to one signature, the three ways of filling it, the checks that run when terms enter,
# and the operations that need the spaces (`H'`, `instantiate`).

const VSU2 = SU2Space(1 // 2 => 1)
const VSU2_0 = SU2Space(0 => 1)
const VU1 = Rep[U₁](0 => 1, 1 => 1)
const VF = Vect[FermionNumber](0 => 1, 1 => 1)

@testset "the four cells of the grid share one signature" begin
    S = spin(VSU2)
    b = dot(S, S)
    fin = FiniteChain(VSU2, 8)
    inf = InfiniteChain(VSU2)
    cells = [
        ("finite", opsum(fin, (b[i] for i in 1:7)), FiniteMPO, [4, 5, 5, 5, 5, 5, 4, 1]),
        ("infinite", opsum(inf, b[1]), InfiniteMPO, [5]),
        (
            "finite + expterm", opsum(fin, b[1], expterm(b[1]; decay = 0.4)), FiniteMPO,
            [7, 5, 5, 5, 5, 5, 4, 1],
        ),
        ("infinite + expterm", opsum(inf, b[1], expterm(b[1]; decay = 0.4)), InfiniteMPO, [8]),
    ]
    for (name, H, T, D) in cells
        @testset "$name" begin
            @test H isa OperatorSum
            mpo = irrep_mpo(H)
            @test mpo isa T
            @test [densedim(mpo.bondsectors, k) for k in eachindex(mpo.bondsectors)] == D
            @test islossless(H)
        end
    end

    H = cells[1][2]
    @test length(H) == 7
    @test H.lattice === fin
    @test isempty(H.channels)
    @test chain_terms(H) ≈ H.terms

    # the channel is one extra bond index on a finite chain, and the plain model is unchanged
    Hd = cells[3][2]
    @test length(Hd) == 1 && length(Hd.channels) == 1
    @test !isempty(Hd)
    # the dense oracle of a decaying model is the geometric sum truncated to the chain
    Hs = opsum(FiniteChain(VSU2, 4), b[1], expterm(b[1]; decay = 0.4))
    @test instantiate(Hs) ≈ instantiate(opsum(FiniteChain(VSU2, 4), chain_terms(Hs)))
    O = instantiate(Hs)
    @test mpo_tensormap(irrep_mpo_tensors(irrep_mpo(Hs), Hs.lattice)) ≈ O

    # an SVD selector is available on a plain finite chain, and nowhere else
    @test irrep_mpo(cells[1][2], SVDBondAlgorithm(truncrank(6))) isa FiniteMPO
    @test_throws ArgumentError irrep_mpo(cells[2][2], SVDBondAlgorithm(truncrank(6)))
    @test_throws ArgumentError irrep_mpo(cells[3][2], SVDBondAlgorithm(truncrank(6)))

    @test length(jordan_mpo_tensors(cells[1][2])) == 8
    @test_throws ArgumentError jordan_mpo_tensors(cells[2][2])
    @test_throws ArgumentError jordan_mpo_tensors(cells[3][2])
end

@testset "opsum!, opsum(lat, …) and H += x agree" begin
    S = spin(VSU2)
    b = dot(S, S)
    N = 6
    lat = FiniteChain(VSU2, N)

    H1 = opsum(lat, (b[i] for i in 1:(N - 1)))
    H2 = OperatorSum(lat)
    @test isempty(H2)
    @test opsum!(H2, (b[i] for i in 1:(N - 1))) === H2
    H3 = OperatorSum(lat)
    for i in 1:(N - 1)
        H3 += b[i]
    end
    H4 = OperatorSum(fill(VSU2, N))                 # any iterable of spaces is a FiniteChain
    opsum!(H4, [b[i] for i in 1:(N - 1)])
    @test H4.lattice isa FiniteChain && H4.lattice == lat
    @test H1 == H2 == H3 == H4
    @test H1 ≈ H3

    # `+` copies, `opsum!` mutates
    H = OperatorSum(lat)
    G = H + b[1]
    @test isempty(H) && length(G) == 1
    opsum!(H, b[1], b[1])                            # coincident terms sum on observation
    @test length(H) == 1
    @test only(H).coeff ≈ 2 * only(b[1]).coeff
    @test H == 2 * G
    @test copy(H) == H && copy(H) !== H

    # nested iterables, bags, single terms and the lattice-first / vector-first spellings
    t = only(b[1])
    @test opsum(lat, [b[1], [b[2], (b[3],)]], t) ≈ opsum(lat, b[1], b[2], b[3], t)
    @test opsum(fill(VSU2, N), b[1]) == opsum(lat, b[1])
    @test isempty(opsum(lat))

    # the latticeless accumulator is still there and still returns a bag
    @test opsum(b[1], b[2]) isa Terms

    # failed insertion leaves the container untouched
    H = opsum(lat, b[1])
    @test_throws ArgumentError opsum!(H, b[2], b[N])      # b[N] reaches site N + 1
    @test length(H) == 1

    # arithmetic
    @test H + H == 2 * H
    @test H - H == zero(H)
    @test isempty((H - H).terms)
    @test -H == (-1) * H
    @test (2H) / 2 == H
    @test H + b[2] == opsum(lat, b[1], b[2])
    @test b[2] + H == H + b[2]
    @test H - b[1] == zero(H)

    # bad arguments are named, not recursed into
    @test_throws ArgumentError opsum!(H, 1.0)
    @test_throws ArgumentError opsum!(H, S)               # an unplaced SiteOperator
    @test_throws ArgumentError opsum!(H, b)               # an unplaced LocalOperator
    @test_throws ArgumentError opsum!(H, H)
end

@testset "H + H' on hopping chains" begin
    # U(1) spin-½: S⁺ᵢ S⁻ᵢ₊₁ + h.c. against a hand-built hermitian operator on 4 sites
    N = 4
    lat = FiniteChain(VU1, N)
    S = spin_ops(VU1, U1Irrep(1), U1Irrep(0))
    T = opsum(lat, couple(S.Sp[i], S.Sm[i + 1]) for i in 1:(N - 1))
    H = T + T'
    @test H isa OperatorSum
    @test length(H) == 2 * (N - 1)
    @test islossless(H)

    # basis (dn, up) = sectors (0, 1) in TensorKit's order: S⁺|dn⟩ = |up⟩
    Sp = [0 0; 1 0]
    Sm = [0 1; 0 0]
    id2 = [1 0; 0 1]
    site(ops) = foldl(kron, [get(ops, i, id2) for i in N:-1:1])
    Hhand = sum(site(Dict(i => Sp, i + 1 => Sm)) + site(Dict(i => Sm, i + 1 => Sp)) for i in 1:(N - 1))
    Hdense = physmatrix(instantiate(H), N, 2)
    @test Hdense ≈ Hhand
    @test Hdense ≈ Hdense'
    @test mpo_tensormap(irrep_mpo_tensors(irrep_mpo(H), lat)) ≈ instantiate(H)

    # fermions: the adjoint supplies the anticommutation sign the caller used to hand-write
    F = fermion_ops(VF)
    flat = FiniteChain(VF, 5)
    Tf = opsum(flat, -1.0 * couple(F.cd[i], F.c[i + 1]) for i in 1:4)
    Hf = Tf + Tf'
    @test Hf ≈ opsum(flat, -1.0 * (couple(F.cd[i], F.c[i + 1]) + couple(F.cd[i + 1], F.c[i])) for i in 1:4)
    @test islossless(Hf)
    @test (Hf')' ≈ Hf && Hf' ≈ Hf

    # infinite: the case TermSum could not reach. T + T' is the same generating set as writing both
    # hops, and the MPO is lossless.
    ilat = InfiniteChain(VF)
    Ti = opsum(ilat, -1.0 * couple(F.cd[1], F.c[2]))
    Hi = Ti + Ti'
    @test Hi isa OperatorSum && Hi.lattice isa InfiniteChain
    @test Hi ≈ opsum(ilat, -1.0 * (couple(F.cd[1], F.c[2]) + couple(F.cd[2], F.c[1])))
    @test islossless(Hi)
    both = irrep_mpo(opsum(ilat, -1.0 * (couple(F.cd[1], F.c[2]) + couple(F.cd[2], F.c[1]))))
    @test irrep_mpo(Hi).bondsectors == both.bondsectors

    # a bare bag has no postfix adjoint, and says where to put it
    @test_throws ArgumentError couple(F.cd[1], F.c[2])'

    # charged terms have no adjoint in the same sector
    Vs = SU2Space(1 // 2 => 1)
    Ss = spin(Vs)
    @test_throws ArgumentError opsum(FiniteChain(Vs, 2), Ss[1])'
end

@testset "the checks that run when terms enter" begin
    S = spin(VSU2)
    b = dot(S, S)
    lat = FiniteChain(VSU2, 4)

    # wrong symmetry sector
    Su1 = spin_ops(VU1, U1Irrep(1), U1Irrep(0))
    err = try
        opsum(lat, couple(Su1.Sz[1], Su1.Sz[2]))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("sector type", err.msg)
    @test_throws ArgumentError opsum(lat, [b[1], couple(Su1.Sz[1], Su1.Sz[2])])

    # a letter the site's space does not carry: a spin-½ operator on a spin-0 site
    mixed = FiniteChain([VSU2, VSU2_0, VSU2])
    err = try
        opsum(mixed, b[1])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("does not exist on site 2", err.msg)
    @test occursin("operator(s) of charge", err.msg)

    # site out of range, on a finite chain only: an infinite chain wraps
    err = try
        opsum(lat, b[4])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("outside the lattice `1:4`", err.msg)
    @test_throws ArgumentError opsum(lat, S[0])
    @test length(opsum(InfiniteChain(VSU2), b[4])) == 1

    # on an infinite chain H is a generating set: neutrality and K = 0 are refused on insertion …
    inf = InfiniteChain(VSU2)
    err = try
        opsum(inf, S[1])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("charge-neutral", err.msg)
    err = try
        opsum(inf, one(Terms{SU2Irrep}))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("K = 0", err.msg)
    charged = expterm(couple(S[1], S[2]; to = SU2Irrep(1)); decay = 0.5)
    @test_throws ArgumentError opsum(inf, charged)
    @test length(opsum(lat, charged).channels) == 1          # fine on a finite chain
    # … a channel's letters are checked against the cell
    @test_throws ArgumentError opsum(
        InfiniteChain(VSU2_0), expterm(b[1]; decay = 0.5)
    )

    # … but two representatives of one translation class are only visible across the whole set, so
    # they are accepted on insertion and refused when the MPO is formed
    Hdup = opsum(inf, b[1], b[2])
    @test length(Hdup) == 2
    err = try
        irrep_mpo(Hdup)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("translates of each other", err.msg)
    @test_throws ArgumentError islossless(Hdup)
    # the same pair is two distinct bonds on a two-site cell
    @test islossless(opsum(InfiniteChain([VSU2, VSU2]), b[1], b[2]))
    # a term written twice is one term, not a duplicate
    @test islossless(opsum(inf, b[1], b[1]))
    twochan = opsum(inf, b[1], expterm(b[1]; decay = 0.5), expterm(b[2]; decay = 0.5))
    @test_throws ArgumentError irrep_mpo(twochan)

    # an empty container is still an empty container
    @test isempty(OperatorSum(lat))
    @test_throws ArgumentError instantiate(OperatorSum(lat))
    @test_throws ArgumentError OperatorSum(3)
    @test_throws ArgumentError OperatorSum(irrep_mpo)
end

@testset "combining operators" begin
    S = spin(VSU2)
    b = dot(S, S)
    a = opsum(FiniteChain(VSU2, 6), b[1])
    c = opsum(FiniteChain(VSU2, 6), b[2])
    @test a + c == opsum(FiniteChain(VSU2, 6), b[1], b[2])
    @test a - c == opsum(FiniteChain(VSU2, 6), b[1], -b[2])

    # different lattices do not mix
    for other in (FiniteChain(VSU2, 5), InfiniteChain(VSU2), FiniteChain(VSU2_0, 6))
        d = OperatorSum(other)
        @test_throws ArgumentError a + d
        @test_throws ArgumentError a - d
    end
    @test a != opsum(FiniteChain(VSU2, 5), b[1])
    @test !(a ≈ opsum(FiniteChain(VSU2, 5), b[1]))

    # channels travel with the container
    e = expterm(b[1]; decay = 0.3)
    ce = opsum(FiniteChain(VSU2, 6), e)
    @test length(ce.channels) == 1 && length(ce) == 0 && !isempty(ce)
    @test (a + ce).channels.channels == ce.channels.channels
    @test (a + e) == (a + ce)
    @test e + a == a + e
    @test isempty((ce - ce).channels)
    @test islossless(ce)
    key = first(keys(ce.channels.channels))
    @test (2 * ce).channels.channels[key] ≈ 2 * ce.channels.channels[key]
    @test ce ≈ ce && ce == copy(ce)
    @test !(ce ≈ a)

    # a bare bag and a bare ExpSum no longer meet outside a container
    @test_throws ArgumentError b[1] + e
    @test_throws ArgumentError e + b[1]

    # sector mismatch
    Su1 = spin_ops(VU1, U1Irrep(1), U1Irrep(0))
    @test_throws ArgumentError a + couple(Su1.Sz[1], Su1.Sz[2])
end

@testset "operations that do not apply" begin
    S = spin(VSU2)
    b = dot(S, S)
    lat = FiniteChain(VSU2, 4)
    H = opsum(lat, b[1])
    Hd = opsum(lat, b[1], expterm(b[1]; decay = 0.4))

    # H' with channels
    err = try
        Hd'
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("channels", err.msg)

    # couple / dot / couple_channels act on building blocks, not on an OperatorSum
    for f in (couple, couple_channels, dot)
        @test_throws ArgumentError f(H, S[1])
        @test_throws ArgumentError f(S[1], H)
    end
    @test_throws ArgumentError couple(H, H)
    @test_throws ArgumentError couple(H, S[1], S[2])
    @test_throws ArgumentError couple(S[1], H, S[2])

    # an infinite chain has no finite operator to materialise
    @test_throws ArgumentError instantiate(opsum(InfiniteChain(VSU2), b[1]))

    # the `(h, lat)` forms are gone; the bag forms say where to go, the rest fail as plain methods
    @test_throws ArgumentError irrep_mpo(b[1], lat)
    @test_throws ArgumentError irrep_mpo(b[1], lat, BipartiteAlgorithm())
    @test_throws ArgumentError irrep_mpo(only(b[1]), lat)
    @test_throws ArgumentError irrep_mpo(expterm(b[1]; decay = 0.4), lat)
    @test_throws MethodError islossless(b[1], lat)
    @test_throws MethodError instantiate(b[1], lat)
    @test_throws MethodError jordan_mpo_tensors(b[1], lat)
    @test_throws MethodError OpSum.adjoint(b[1], lat)
    @test !isdefined(OpSum, :MixedSum)

    # show prints the lattice compactly and does not fail with channels
    @test occursin("FiniteChain", sprint(show, H))
    @test occursin("ExpSum", sprint(show, Hd))
    @test occursin("InfiniteChain", sprint(show, opsum(InfiniteChain(VSU2), b[1])))
end

@testset "the unit-cell view of an infinite OperatorSum" begin
    S = spin(VSU2)
    b = dot(S, S)
    inf = InfiniteChain(VSU2)
    H = opsum(inf, b[3], expterm(b[2]; decay = 0.4))
    G = unitcell_terms(H)
    @test G isa OperatorSum && G.lattice === inf
    @test only(G).sites == [1, 2]
    @test only(keys(G.channels.channels)).term.sites == [1, 2]
    # translation does not change the MPO
    @test irrep_mpo(H).bondsectors == irrep_mpo(opsum(inf, b[1], expterm(b[1]; decay = 0.4))).bondsectors
    @test OpSum.maxspan(G) == 1
    @test islossless(G)
end
