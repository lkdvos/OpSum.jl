using Test
using OpSum
using OpSum: LocalOperator, nsites, project, instantiate, couple, couple_channels, spin, spin_ops,
    fermion_ops, scalarop, opsum, Terms, FiniteChain
using TensorKit
using TensorKit: removeunit, numind
using LinearAlgebra: dot

# `instantiate` output as the plain `V ← V` map (trailing trivial charge leg dropped)
function densop(h, lat)
    O = instantiate(opsum(lat, h))
    return numind(O) == 2 * numout(O) ? O : removeunit(O, numind(O))
end

const Vsu2 = SU2Space(1 // 2 => 1)
const Vu1 = Rep[U₁](0 => 1, 1 => 1)
const Vf = Vect[FermionNumber](0 => 1, 1 => 1)
const S = spin(Vsu2)
const Sp, Sm, Sz = spin_ops(Vu1, U1Irrep(1), U1Irrep(0))
const F = fermion_ops(Vf)

# contiguous and gapped placements of a K-slot block
placements(K) = (collect(1:K), collect(3:(K + 2)), collect(1:2:(2K - 1)))

@testset "placement is project(h, sites), also across a gap" begin
    xxz(i, j) = couple(Sp[i], Sm[j]) / 2 + couple(Sm[i], Sp[j]) / 2 + 0.7 * couple(Sz[i], Sz[j])
    hop(i, j) = -(couple(F.cd[i], F.c[j]) + couple(F.cd[j], F.c[i]))
    cases = (
        (Vsu2, dot(S[1], S[2]), (i, j) -> dot(S[i], S[j])),
        (Vu1, xxz(1, 2), xxz),
        (Vf, hop(1, 2), hop),
    )
    for (V, h2, direct) in cases
        B = project(instantiate(opsum([V, V], h2)))
        @test B isa LocalOperator && nsites(B) == 2
        for sites in ([1, 2], [3, 4], [1, 3], [2, 7])
            @test B[sites...] == project(instantiate(opsum([V, V], h2)), sites)
        end
        @test B[4] == B[4, 5]
        # the gap site is a pass-through; the dense operator is the directly placed one
        for (i, j, N) in ((1, 3, 3), (2, 4, 4), (1, 4, 4))
            @test B[i, j] ≈ direct(i, j)
            @test densop(B[i, j], FiniteChain(V, N)) ≈ densop(direct(i, j), FiniteChain(V, N))
        end
    end
    # K = 3 with a charged intermediate
    chi = couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(1))
    B3 = project(instantiate(opsum(fill(Vsu2, 3), chi)))
    @test nsites(B3) == 3
    @test B3[1, 4, 5] ≈ couple(couple(S[1], S[4]; to = SU2Irrep(1)), S[5]; to = SU2Irrep(1))
end

@testset "unplaced couple/dot/couple_channels match the placed ones" begin
    for sites in placements(2)
        i, j = sites
        @test dot(S, S)[i, j] == dot(S[i], S[j])
        @test couple(Sp, Sm)[i, j] == couple(Sp[i], Sm[j])
        @test couple(F.cd, F.c)[i, j] == couple(F.cd[i], F.c[j])
        @test couple(S, S; to = SU2Irrep(1))[i, j] == couple(S[i], S[j]; to = SU2Irrep(1))
    end
    # never reordered: the out-of-order placed spelling differs by the anticommutation sign
    @test couple(F.c, F.cd)[1, 2] ≈ -couple(F.cd[2], F.c[1])
    for (i, j, k, l) in placements(4)
        @test couple(F.cd, F.c, F.cd, F.c)[i, j, k, l] == couple(F.cd[i], F.c[j], F.cd[k], F.c[l])
    end
    @test couple(Sp, Sm, Sz)[1, 3, 5] == couple(Sp[1], Sm[3], Sz[5])
    @test couple(S, S, S)[2, 3, 5] == couple(S[2], S[3], S[5])
    # SU(2) channels: forced (to = 3 needs j₁₂ = 2) and named, plus the query
    @test couple(S, S, S; to = SU2Irrep(3))[1] == couple(S[1], S[2], S[3]; to = SU2Irrep(3))
    @test couple_channels(S, S, S, S) == couple_channels(S[1], S[2], S[3], S[4])
    for (j12, j123) in couple_channels(S, S, S, S)
        T = couple(couple(couple(S, S; to = j12), S; to = j123), S)
        @test nsites(T) == 4
        @test T[1, 3, 4, 6] == couple(couple(couple(S[1], S[3]; to = j12), S[4]; to = j123), S[6])
    end
    @test couple_channels(F.cd, F.c, F.cd, F.c) == couple_channels(F.cd[1], F.c[2], F.cd[3], F.c[4])
end

@testset "pass-through slots" begin
    # B ± α: the all-pass-through term, dropped on placement
    b = dot(S, S)
    I0 = Terms{SU2Irrep}
    @test (b + 1 / 4)[1] == dot(S[1], S[2]) + one(I0) / 4
    @test (1 / 4 - b)[2, 5] ≈ one(I0) / 4 - dot(S[2], S[5])
    @test nsites(b - 1 / 4) == 2

    # a scalar part of an operand becomes a pass-through slot
    @test couple(Sz + scalarop(0.5, Vu1), Sz)[2, 4] ≈ couple(Sz[2], Sz[4]) + Sz[4] / 2
    @test LocalOperator(scalarop(2.0, Vsu2))[3] == scalarop(2.0, Vsu2)[3]

    # inside a charged caterpillar the idle slot carries the running bond charge, contiguous and gapped
    α = 0.3
    B = couple(Sp, Sz + scalarop(α, Vu1), Sm)
    ref(i, j, k) = couple(Sp[i], Sz[j], Sm[k]) + α * couple(Sp[i], Sm[k])
    for sites in ([1, 2, 3], [1, 3, 4], [2, 3, 5])
        @test B[sites...] ≈ ref(sites...)
        lat = FiniteChain(Vu1, maximum(sites))
        @test densop(B[sites...], lat) ≈ densop(ref(sites...), lat)
    end
    β = 0.5
    Bf = couple(F.cd, F.n + scalarop(β, Vf), F.c)
    reff(i, j, k) = couple(F.cd[i], F.n[j], F.c[k]) + β * couple(F.cd[i], F.c[k])
    for sites in ([1, 2, 3], [1, 3, 4], [1, 2, 4])
        @test Bf[sites...] ≈ reff(sites...)
        lat = FiniteChain(Vf, maximum(sites))
        @test densop(Bf[sites...], lat) ≈ densop(reff(sites...), lat)
    end
end

@testset "errors" begin
    B = project(instantiate(opsum([Vsu2, Vsu2], dot(S[1], S[2]))))
    @test_throws ArgumentError B[1, 2, 3]         # wrong index count
    @test_throws ArgumentError B[2, 1]            # non-increasing
    @test_throws ArgumentError B[0]               # site < 1
    @test_throws ArgumentError LocalOperator(dot(S[1], S[3]), 2)     # support is not 1:K
    @test_throws ArgumentError B + project(instantiate(opsum(fill(Vsu2, 3), couple(S[1], S[2], S[3]))))
    @test_throws ArgumentError couple(S, couple(S, S; to = SU2Irrep(1)))    # multi-slot later operand
    @test_throws MethodError couple(S, S[2])      # placed and unplaced do not mix
    @test_throws ArgumentError couple(S, S, S, S) # unforced channel, as for the placed form
    @test B + B ≈ 2 * B
end
