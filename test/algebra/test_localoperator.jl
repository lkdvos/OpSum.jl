using Test
using OpSum
using OpSum: LocalOperator, nsites, project, instantiate, couple, couple_channels, spin_ops,
    spin_basis, scalarop, opsum, Terms, FiniteChain, SiteOperator
using TensorKit
using LinearAlgebra: dot

include(joinpath(@__DIR__, "..", "testutils.jl"))   # unitbasis, spinvector, matrixunit, fermionmode

densop(h, lat) = plain(instantiate(opsum(lat, h)))

# SU(2) and fermions use matrix-unit bases; U(1) spin-½ the stock basis, where `S_z` is one atomic letter.
const Vsu2 = unitbasis(SU2Space(1 // 2 => 1))
const Vu1 = spin_basis(U1Irrep)
const Vf = unitbasis(Vect[FermionNumber](0 => 1, 1 => 1))
const S = spinvector(Vsu2)
const Sp = matrixunit(Vu1, U1Irrep(1 // 2), U1Irrep(-1 // 2))
const Sm = matrixunit(Vu1, U1Irrep(-1 // 2), U1Irrep(1 // 2))
const Sz = spin_ops(U1Irrep).S_z
const F = fermionmode(Vf, FermionNumber(0), FermionNumber(1))

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
        B = project(instantiate(opsum(FiniteChain(V, 2), h2)), V)
        @test B isa LocalOperator && nsites(B) == 2
        @test B[4] == B[4, 5]
        # the gap site is a pass-through; the dense operator is the directly placed one
        for (i, j, N) in ((1, 3, 3), (2, 4, 4), (1, 4, 4))
            @test B[i, j] ≈ direct(i, j)
            @test densop(B[i, j], FiniteChain(V, N)) ≈ densop(direct(i, j), FiniteChain(V, N))
        end
    end
    # K = 3 with a charged intermediate
    chi = couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(1))
    B3 = project(instantiate(opsum(FiniteChain(Vsu2, 3), chi)), Vsu2)
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
    b = dot(S, S)
    I0 = Terms{SU2Irrep}
    @test (b + 1 / 4)[1] == dot(S[1], S[2]) + one(I0) / 4
    @test (1 / 4 - b)[2, 5] ≈ one(I0) / 4 - dot(S[2], S[5])
    @test nsites(b - 1 / 4) == 2
    @test LocalOperator(scalarop(2.0, Vsu2))[3] == scalarop(2.0, Vsu2)[3]

    # inside a charged caterpillar the idle slot of a LocalOperator carries the running bond charge
    B = project(instantiate(opsum(FiniteChain(Vu1, 2), couple(Sp[1], Sm[2]))), Vu1)   # S⁺ ⊗ S⁻
    A = couple(B, Sz; to = unit(U1Irrep))
    @test nsites(A) == 3
    for sites in ([1, 2, 3], [1, 3, 4], [2, 3, 5])
        @test A[sites...] ≈ couple(couple(Sp[sites[1]], Sm[sites[2]]), Sz[sites[3]]; to = unit(U1Irrep))
    end
end

@testset "scalars are not coupled" begin
    Sz1 = Sz + scalarop(0.5, Vu1)
    # unplaced couple, couple_channels and dot reject a scalar part, as the placed forms do
    @test_throws ArgumentError couple(Sz[1] + scalarop(0.5, Vu1)[1], Sz[2])
    @test_throws ArgumentError couple(Sz, Sz1)
    @test_throws ArgumentError couple(Sz1, Sz)
    @test_throws ArgumentError couple(Sz, Sz1, Sz)
    @test_throws ArgumentError couple(Sz, Sz, Sz1; to = unit(U1Irrep))
    @test_throws ArgumentError couple_channels(Sz, Sz1, Sz)
    @test_throws ArgumentError couple_channels(Sz1, Sz)
    @test_throws ArgumentError dot(Sz, Sz1)
    @test_throws ArgumentError dot(Sz1, Sz)
    @test_throws ArgumentError couple(scalarop(2.0, Vsu2), S)
    @test_throws ArgumentError couple(S, scalarop(2.0, Vsu2))
    @test_throws ArgumentError couple(one(dot(S, S)), S)
    @test_throws ArgumentError couple(dot(S, S) + 1, S)
    # the unplaced error is the placed one
    err(f) = try
        f()
        nothing
    catch e
        e.msg
    end
    @test err(() -> couple(Sz1, Sz)) == err(() -> couple(Sz1[1], Sz[2]))
    @test err(() -> couple(Sz, Sz1)) == err(() -> couple(Sz[1], Sz1[2]))
    @test err(() -> couple_channels(Sz, Sz1, Sz)) == err(() -> couple_channels(Sz[1], Sz1[2], Sz[3]))
    @test err(() -> dot(Sz, Sz1)) == err(() -> dot(Sz[1], Sz1[2]))
end

@testset "the identity stays an operator" begin
    for (V, N) in ((Vsu2, 3), (Vu1, 3), (Vf, 2))
        lat = FiniteChain(V, N)
        idN = id(foldl(⊗, (space(V) for _ in 1:N)))
        I1 = sectortype(V)
        # SiteOperator: one(op), one(type), scalarop(c, basis | sector)
        @test isone(one(SiteOperator{I1})) && isone(one(scalarop(2, V)))
        @test one(SiteOperator{I1})[2] == one(Terms{I1}) == scalarop(1, V)[N]
        @test densop(one(SiteOperator{I1})[2], lat) ≈ idN
        @test densop(scalarop(2.5, V)[1], lat) ≈ 2.5 * idN
        @test densop(scalarop(2.5, I1)[N], lat) ≈ 2.5 * idN
        # Terms: one(Terms), also scaled
        @test densop(one(Terms{I1}), lat) ≈ idN
        @test densop(3 * one(Terms{I1}), lat) ≈ 3 * idN
        # LocalOperator: one(B), B ± α (one slot here, two below)
        B = LocalOperator(scalarop(1.0, V))
        @test nsites(one(B)) == 1
        @test densop(one(B)[2], lat) ≈ idN
        @test one(B) + 2.0 ≈ 3.0 * one(B)
        @test densop((one(B) + 2.0)[1], lat) ≈ 3 * idN
    end
    # a two-slot operator: B + α is B plus α·id on the lattice, and one(B) is the identity on its slots
    B2 = dot(S, S)
    lat = FiniteChain(Vsu2, 3)
    idN = id(foldl(⊗, (space(Vsu2) for _ in 1:3)))
    @test nsites(one(B2)) == 2
    @test densop(one(B2)[2], lat) ≈ idN
    @test densop((B2 + 0.25)[1, 3], lat) ≈ densop(B2[1, 3], lat) + 0.25 * idN
    @test densop((0.25 - B2)[2], lat) ≈ 0.25 * idN - densop(B2[2], lat)
end

@testset "errors" begin
    B = project(instantiate(opsum(FiniteChain(Vsu2, 2), dot(S[1], S[2]))), Vsu2)
    @test_throws ArgumentError B[1, 2, 3]         # wrong index count
    @test B[2, 1] ≈ B[1, 2]                      # out of order: re-coupled, and S⃗₂·S⃗₁ = S⃗₁·S⃗₂
    @test_throws ArgumentError B[0]               # site < 1
    @test_throws ArgumentError LocalOperator(dot(S[1], S[3]), 2)     # support is not 1:K
    @test_throws ArgumentError B + project(instantiate(opsum(FiniteChain(Vsu2, 3), couple(S[1], S[2], S[3]))), Vsu2)
    @test_throws ArgumentError couple(S, couple(S, S; to = SU2Irrep(1)))    # multi-slot later operand
    @test_throws MethodError couple(S, S[2])      # placed and unplaced do not mix
    @test_throws ArgumentError couple(S, S, S, S) # unforced channel, as for the placed form
end
