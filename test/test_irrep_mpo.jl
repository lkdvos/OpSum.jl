using Test
using OpSum
using OpSum: irrep_mpo, mpo_terms, TermSum, spin, scalarop, couple, opsum
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot

include(joinpath(@__DIR__, "testutils.jl"))   # LO, islossless, densedim

@testset "SU(2) Heisenberg — lossless + per-sector bond dims" begin
    V = SU2Space(1 // 2 => 1)
    for N in (3, 4, 5)
        sites = fill(V, N)
        H = opsum(sites, (dot(spin(V)[i], spin(V)[i + 1]) for i in 1:(N - 1)))
        @test islossless(H)

        Ws, secs = irrep_mpo(H)
        # right boundary is the trivial (singlet) total charge
        @test secs[N] == [SU2Irrep(0)]
        # every bond charge is 0 or 1 (nearest-neighbour spin coupling)
        @test all(all(c -> c in (SU2Irrep(0), SU2Irrep(1)), s) for s in secs)
        # coupling charge propagates on internal bonds
        @test any(SU2Irrep(1) in s for s in secs[1:(N - 1)])
    end

    # bulk (deep-interior) bond of a longer chain: canonical Heisenberg structure
    N = 6
    V = SU2Space(1 // 2 => 1)
    _, secs = irrep_mpo(
        opsum(fill(V, N), (dot(spin(V)[i], spin(V)[i + 1]) for i in 1:(N - 1)))
    )
    b = 3    # a bulk bond
    @test count(==(SU2Irrep(0)), secs[b]) == 2      # identity-in + identity-out
    @test count(==(SU2Irrep(1)), secs[b]) == 1      # one open spin multiplet
    @test densedim(secs, b) == 5                    # 1·2 + 3·1 = 5 (dense Heisenberg bond dim)
end

@testset "U(1) hopping — lossless + charge-resolved bonds" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    raise = LO(IrrepOperator(U1Irrep(1), 1))
    lower = LO(IrrepOperator(U1Irrep(-1), 1))
    N = 4
    sites = fill(V, N)
    # number-conserving hopping Σ (b†_i b_{i+1} + h.c.)  (built as raise·lower both directions)
    H = opsum(
        sites,
        (dot(raise[i], lower[i + 1]) for i in 1:(N - 1)),
        (dot(lower[i], raise[i + 1]) for i in 1:(N - 1)),
    )
    @test islossless(H)

    _, secs = irrep_mpo(H)
    @test secs[N] == [U1Irrep(0)]
    # internal bonds carry the flowing charges 0 and ±1
    @test all(all(c -> c in (U1Irrep(0), U1Irrep(1), U1Irrep(-1)), s) for s in secs)
    @test any(U1Irrep(1) in s || U1Irrep(-1) in s for s in secs[1:(N - 1)])
end

@testset "trivial sector (ℂ²) — lossless" begin
    V = ℂ^2
    ops = instances(IrrepOperator, V)
    N = 3
    sites = fill(V, N)
    # a chain of nearest-neighbour couplings of two nontrivial trivial-sector letters
    H = opsum(
        sites,
        (couple(LO(ops[2])[i], LO(ops[3])[i + 1]; to = unit(Trivial)) for i in 1:(N - 1))
    )
    @test islossless(H)
    _, secs = irrep_mpo(H)
    @test all(all(==(unit(Trivial)), s) for s in secs)   # only the trivial charge exists
end

@testset "scalar + coupling (charge-0 mix)" begin
    V = SU2Space(1 // 2 => 1)
    N = 3
    sites = fill(V, N)
    # a constant plus a coupling — both total charge 0
    H = opsum(sites, scalarop(2.5, V)[1], dot(spin(V)[1], spin(V)[2]))
    @test islossless(H)
    _, secs = irrep_mpo(H)
    @test secs[N] == [SU2Irrep(0)]
end

@testset "sum of single-site fields (uniform charge)" begin
    V = SU2Space(1 // 2 => 1)
    N = 3
    sites = fill(V, N)
    H = opsum(sites, spin(V)[1], spin(V)[2], spin(V)[3])   # each term total charge 1
    @test islossless(H)
    _, secs = irrep_mpo(H)
    @test secs[N] == [SU2Irrep(1)]
end

@testset "K=3 SU(2) three-body scalar — lossless" begin
    V = SU2Space(1 // 2 => 1)
    S = spin(V)
    sites = fill(V, 3)
    H = opsum(sites, couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0)))
    @test islossless(H)
    _, secs = irrep_mpo(H)
    @test secs[3] == [SU2Irrep(0)]
    # the internal bonds carry the caterpillar inner line (spin-1 after the first pair)
    @test SU2Irrep(1) in secs[2]
end

@testset "decoupled singlet pairs — multiple same-charge disconnected states at a bond" begin
    # three independent nearest-neighbour singlet pairs on a 6-site chain: at bulk bonds, several
    # mutually-disconnected trivial-charge states coexist ("nothing started yet" vs "already closed
    # a pair to a singlet"), exercising the per-connected-component vertex-cover split within a
    # single bond-charge sector (not just across sectors).
    V = SU2Space(1 // 2 => 1)
    N = 6
    sites = fill(V, N)
    H = opsum(
        sites,
        dot(spin(V)[1], spin(V)[2]),
        dot(spin(V)[3], spin(V)[4]),
        dot(spin(V)[5], spin(V)[6]),
    )
    @test islossless(H)

    _, secs = irrep_mpo(H)
    @test secs[N] == [SU2Irrep(0)]
    @test count(==(SU2Irrep(0)), secs[2]) == 2      # "identity so far" + "first pair already closed"
    @test count(==(SU2Irrep(0)), secs[3]) == 2
    @test count(==(SU2Irrep(1)), secs[3]) == 1      # the second pair's open coupling
end

@testset "K=3 U(1) three-body term — lossless + charge-resolved" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    raise = LO(IrrepOperator(U1Irrep(1), 1))
    lower = LO(IrrepOperator(U1Irrep(-1), 1))
    sites = fill(V, 3)
    # charges (1, 1, -1) fusing through inner line 2 to a net charge 1
    H = opsum(sites, couple(couple(raise[1], raise[2]; to = U1Irrep(2)), lower[3]; to = U1Irrep(1)))
    @test islossless(H)
    _, secs = irrep_mpo(H)
    @test secs[3] == [U1Irrep(1)]
    @test U1Irrep(2) in secs[2]           # inner-line charge on the bond after site 2
end
