using Test
using OpSum
using OpSum: mpo, reduced_tensors, reduced_terms, instantiate, scalarop, couple, opsum
using OpSum: VertexCover, IndependentSVD, SequentialSVD
using OpSum: Letter
using TensorKit
using TensorKit: @tensor
using LinearAlgebra: dot
using MatrixAlgebraKit: MatrixAlgebraKit, truncrank, NoTruncation

include(joinpath(@__DIR__, "..", "testutils.jl"))   # LO, physmatrix, densedim

function contract3(T)
    return @tensor Op[o1 o2 o3 bL; bR i1 i2 i3] :=
        T[1][bL o1; i1 b1] * T[2][b1 o2; i2 b2] * T[3][b2 o3; i3 bR]
end

# operator the reduced MPO represents, via path enumeration (valid for any rotated/truncated bond basis)
mpo_operator(Ws, secs, sites) = instantiate(opsum(sites, reduced_terms(Ws, secs)))

@testset "lossless SVD reproduces the operator and matches the bipartite bond dimensions" begin
    Vs = SU2Space(1 // 2 => 1)
    S = spinvector(unitbasis(Vs))
    Vu = Rep[U₁](0 => 1, 1 => 1)
    raise, lower = LO(Letter(U1Irrep(1), 1)), LO(Letter(U1Irrep(-1), 1))
    ops = [LO(Letter(unit(Trivial), n)) for n in 1:4]
    cases = Any[
        ("SU2 Heisenberg N=$N", unitbasis(Vs), N, Terms(dot(S[i], S[i + 1]) for i in 1:(N - 1)), SU2Irrep(0)) for N in (3, 4, 5)
    ]
    push!(
        cases,
        (
            "U1 hopping", unitbasis(Vu), 4,
            Terms((dot(raise[i], lower[i + 1]) for i in 1:3), (dot(lower[i], raise[i + 1]) for i in 1:3)), U1Irrep(0),
        ),
        (
            "trivial", unitbasis(ℂ^2), 3,
            Terms(couple(ops[2][i], ops[3][i + 1]; to = unit(Trivial)) for i in 1:2), unit(Trivial),
        ),
        ("SU2 K=3", unitbasis(Vs), 3, Terms(couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))), SU2Irrep(0)),
    )
    for (name, B, N, H, total) in cases
        @testset "$name" begin
            sites = fill(B, N)
            Wb, sb = reduced(mpo(opsum(sites, H), VertexCover()))
            Ws, ss = reduced(mpo(opsum(sites, H), IndependentSVD()))
            @test mpo_operator(Ws, ss, sites) ≈ instantiate(opsum(sites, H))
            @test ss[N] == [total]
            @test all(densedim(ss, b) <= densedim(sb, b) for b in 1:N)
            startswith(name, "SU2 Heis") && @test all(densedim(ss, b) == densedim(sb, b) for b in 1:N)
            N == 3 && startswith(name, "SU2 ") && @test SU2Irrep(1) in ss[2]
        end
    end
end

@testset "truncation shrinks the bond globally across sectors" begin
    V = SU2Space(1 // 2 => 1)
    S = spinvector(unitbasis(V))
    N = 3
    sites = fill(unitbasis(V), N)
    H = Terms(dot(S[i], S[i + 1]) for i in 1:(N - 1))

    _, sb = reduced(mpo(opsum(sites, H), VertexCover()))
    _, sfull = reduced(mpo(opsum(sites, H), IndependentSVD()))
    _, sbig = reduced(mpo(opsum(sites, H), IndependentSVD(truncrank(100))))
    @test [length(s) for s in sbig] == [length(s) for s in sfull]

    Wt, st = reduced(mpo(opsum(sites, H), IndependentSVD(truncrank(1))))
    @test densedim(st, 1) < densedim(sb, 1)
    @test length(st[1]) == 1

    Mtrunc = physmatrix(contract3(reduced_tensors(Wt, st, FiniteChain(sites))), N, 2)
    Mexact = physmatrix(instantiate(opsum(sites, H)), N, 2)
    @test size(Mtrunc) == size(Mexact)
    @test !(Mtrunc ≈ Mexact)
end

@testset "sweep selector: the two truncation semantics" begin
    V = SU2Space(1 // 2 => 1)
    S = spinvector(unitbasis(V))
    N = 4
    sites = fill(unitbasis(V), N)
    H = Terms(dot(S[i], S[i + 1]) for i in 1:(N - 1))

    @test IndependentSVD().trunc isa NoTruncation
    @test SequentialSVD().trunc isa NoTruncation
    @test IndependentSVD(truncrank(2)).trunc isa MatrixAlgebraKit.TruncationStrategy

    # `truncrank(1)` is one index per bond for the independent sweep, whereas the sequential sweep
    # truncates in the basis it was handed, so an early truncation starves every bond downstream.
    _, sti = reduced(mpo(opsum(sites, H), IndependentSVD(truncrank(1))))
    @test all(length(s) == 1 for s in sti)
    _, sts = reduced(mpo(opsum(sites, H), SequentialSVD(truncrank(1))))
    @test [length(s) for s in sts] != [length(s) for s in sti]
    @test all(length(s) <= 1 for s in sts)
end
