using Test
using OpSum
using OpSum: irrep_mpo, irrep_mpo_tensors, mpo_terms, instantiate, TermSum, spin, scalarop, couple
using OpSum: BipartiteAlgorithm, SVDBondAlgorithm
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using TensorKit: @tensor
using LinearAlgebra: dot
using MatrixAlgebraKit: truncrank, trunctol

include(joinpath(@__DIR__, "testutils.jl"))   # LO, physmatrix, densedim

contract2(T) = @tensor Op[o1 o2 bL; bR i1 i2] := T[1][bL o1; i1 bm] * T[2][bm o2; i2 bR]
function contract3(T)
    return @tensor Op[o1 o2 o3 bL; bR i1 i2 i3] :=
        T[1][bL o1; i1 b1] * T[2][b1 o2; i2 b2] * T[3][b2 o3; i3 bR]
end

# operator the reduced MPO represents, via the path-enumeration reconstruction (works for any valid
# (Ws, bondsectors), including the SVD-rotated / truncated bond bases).
mpo_operator(Ws, secs, sites) = instantiate(mpo_terms(Ws, secs), sites)

@testset "lossless SVD reproduces the operator (explicit contraction)" begin
    V = SU2Space(1 // 2 => 1)
    d = 2

    H2 = dot(spin(V)[1], spin(V)[2])
    Ws, secs = irrep_mpo(H2, [V, V], SVDBondAlgorithm())
    T = irrep_mpo_tensors(Ws, secs, [V, V])
    @test physmatrix(contract2(T), 2, d) ≈ physmatrix(instantiate(H2, [V, V]), 2, d)

    H3 = dot(spin(V)[1], spin(V)[2]) + dot(spin(V)[2], spin(V)[3])
    Ws3, secs3 = irrep_mpo(H3, fill(V, 3), SVDBondAlgorithm())
    T3 = irrep_mpo_tensors(Ws3, secs3, fill(V, 3))
    @test physmatrix(contract3(T3), 3, d) ≈ physmatrix(instantiate(H3, fill(V, 3)), 3, d)
end

@testset "lossless SVD matches the operator + is no larger than bipartite" begin
    # SVD keeps the exact numerical bond rank; the bipartite min-vertex-cover is an upper bound.
    V = SU2Space(1 // 2 => 1)
    for N in (3, 4, 5)
        sites = fill(V, N)
        H = reduce(+, dot(spin(V)[i], spin(V)[i + 1]) for i in 1:(N - 1))

        Wb, sb = irrep_mpo(H, sites, BipartiteAlgorithm())
        Ws, ss = irrep_mpo(H, sites, SVDBondAlgorithm())

        # same represented operator as the exact bipartite build
        @test mpo_operator(Ws, ss, sites) ≈ mpo_operator(Wb, sb, sites)
        # right boundary is the singlet total charge
        @test ss[N] == [SU2Irrep(0)]
        # lossless SVD bond dims never exceed the bipartite ones (here: equal, Heisenberg)
        @test all(densedim(ss, b) <= densedim(sb, b) for b in 1:N)
        @test all(densedim(ss, b) == densedim(sb, b) for b in 1:N)
    end
end

@testset "lossless SVD — U(1) hopping + trivial sector" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    raise = LO(IrrepOperator(U1Irrep(1), 1))
    lower = LO(IrrepOperator(U1Irrep(-1), 1))
    N = 4
    sites = fill(V, N)
    H = reduce(+, dot(raise[i], lower[i + 1]) for i in 1:(N - 1)) +
        reduce(+, dot(lower[i], raise[i + 1]) for i in 1:(N - 1))
    Ws, ss = irrep_mpo(H, sites, SVDBondAlgorithm())
    Wb, sb = irrep_mpo(H, sites, BipartiteAlgorithm())
    @test mpo_operator(Ws, ss, sites) ≈ mpo_operator(Wb, sb, sites)
    @test ss[N] == [U1Irrep(0)]

    Vt = ℂ^2
    ops = instances(IrrepOperator, Vt)
    st = fill(Vt, 3)
    Ht = reduce(+, couple(LO(ops[2])[i], LO(ops[3])[i + 1]; to = unit(Trivial)) for i in 1:2)
    Wst, sst = irrep_mpo(Ht, st, SVDBondAlgorithm())
    @test mpo_operator(Wst, sst, st) ≈ instantiate(Ht, st)
    @test all(all(==(unit(Trivial)), s) for s in sst)
end

@testset "lossless SVD — K=3 SU(2) three-body scalar" begin
    V = SU2Space(1 // 2 => 1)
    S = spin(V)
    sites = fill(V, 3)
    H = couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))
    Ws, ss = irrep_mpo(H, sites, SVDBondAlgorithm())
    @test mpo_operator(Ws, ss, sites) ≈ instantiate(H, sites)
    @test ss[3] == [SU2Irrep(0)]
    @test SU2Irrep(1) in ss[2]      # caterpillar inner line (spin-1 after the first pair)
end

@testset "truncation shrinks the bond globally across sectors" begin
    V = SU2Space(1 // 2 => 1)
    N = 3
    sites = fill(V, N)
    H = reduce(+, dot(spin(V)[i], spin(V)[i + 1]) for i in 1:(N - 1))

    _, sb = irrep_mpo(H, sites, BipartiteAlgorithm())
    _, sfull = irrep_mpo(H, sites, SVDBondAlgorithm())

    # a generous rank keeps everything → identical to the lossless build
    _, sbig = irrep_mpo(H, sites, SVDBondAlgorithm(truncrank(100)))
    @test [length(s) for s in sbig] == [length(s) for s in sfull]

    # rank-1 per bond forces a strictly smaller interior bond (a lossy approximation)
    Wt, st = irrep_mpo(H, sites, SVDBondAlgorithm(truncrank(1)))
    @test densedim(st, 1) < densedim(sb, 1)
    @test length(st[1]) == 1          # a single retained bond index

    # the truncated tensors still contract to an operator (via the assembled TensorMaps —
    # unlike mpo_terms, this needs no intact identity backbone), but no longer the exact one
    Tt = irrep_mpo_tensors(Wt, st, sites)
    Mtrunc = physmatrix(contract3(Tt), N, 2)
    Mexact = physmatrix(instantiate(H, sites), N, 2)
    @test size(Mtrunc) == size(Mexact)
    @test !(Mtrunc ≈ Mexact)
end
