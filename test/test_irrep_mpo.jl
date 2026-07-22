using Test
using OpSum
using OpSum: irrep_mpo, mpo_terms, TermSum, spin, scalarop, couple
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot

LO(x) = OpSum.LocalOp(x)

# faithfulness: the compressed MPO reconstructs the original term-sum exactly (lossless)
function islossless(H::TermSum, sites)
    Ws, secs = irrep_mpo(H, sites)
    back = mpo_terms(Ws, secs)
    Set(keys(back.terms)) == Set(keys(H.terms)) || return false
    return all(back.terms[k] ≈ H.terms[k] for k in keys(H.terms))
end

# dense-equivalent bond dimension at bond b (qdim-weighted sum of the per-sector multiplicities)
densedim(secs, b) = sum(dim(c) for c in secs[b])

@testset "SU(2) Heisenberg — lossless + per-sector bond dims" begin
    V = SU2Space(1 // 2 => 1)
    for N in (3, 4, 5)
        sites = fill(V, N)
        H = reduce(+, dot(spin(V)[i], spin(V)[i + 1]) for i in 1:(N - 1))
        @test islossless(H, sites)

        Ws, secs = irrep_mpo(H, sites)
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
    _, secs = irrep_mpo(reduce(+, dot(spin(V)[i], spin(V)[i + 1]) for i in 1:(N - 1)), fill(V, N))
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
    H = reduce(+, dot(raise[i], lower[i + 1]) for i in 1:(N - 1)) +
        reduce(+, dot(lower[i], raise[i + 1]) for i in 1:(N - 1))
    @test islossless(H, sites)

    _, secs = irrep_mpo(H, sites)
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
    H = reduce(+, couple(LO(ops[2])[i], LO(ops[3])[i + 1]; to = unit(Trivial)) for i in 1:(N - 1))
    @test islossless(H, sites)
    _, secs = irrep_mpo(H, sites)
    @test all(all(==(unit(Trivial)), s) for s in secs)   # only the trivial charge exists
end

@testset "scalar + coupling (charge-0 mix)" begin
    V = SU2Space(1 // 2 => 1)
    N = 3
    sites = fill(V, N)
    # a constant plus a coupling — both total charge 0
    H = scalarop(2.5, V)[1] + dot(spin(V)[1], spin(V)[2])
    @test islossless(H, sites)
    _, secs = irrep_mpo(H, sites)
    @test secs[N] == [SU2Irrep(0)]
end

@testset "sum of single-site fields (uniform charge)" begin
    V = SU2Space(1 // 2 => 1)
    N = 3
    sites = fill(V, N)
    H = spin(V)[1] + spin(V)[2] + spin(V)[3]     # each term total charge 1
    @test islossless(H, sites)
    _, secs = irrep_mpo(H, sites)
    @test secs[N] == [SU2Irrep(1)]
end

@testset "K=3 SU(2) three-body scalar — lossless" begin
    V = SU2Space(1 // 2 => 1)
    S = spin(V)
    sites = fill(V, 3)
    H = couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))
    @test islossless(H, sites)
    _, secs = irrep_mpo(H, sites)
    @test secs[3] == [SU2Irrep(0)]
    # the internal bonds carry the caterpillar inner line (spin-1 after the first pair)
    @test SU2Irrep(1) in secs[2]
end

@testset "K=3 U(1) three-body term — lossless + charge-resolved" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    raise = LO(IrrepOperator(U1Irrep(1), 1))
    lower = LO(IrrepOperator(U1Irrep(-1), 1))
    sites = fill(V, 3)
    # charges (1, 1, -1) fusing through inner line 2 to a net charge 1
    H = couple(couple(raise[1], raise[2]; to = U1Irrep(2)), lower[3]; to = U1Irrep(1))
    @test islossless(H, sites)
    _, secs = irrep_mpo(H, sites)
    @test secs[3] == [U1Irrep(1)]
    @test U1Irrep(2) in secs[2]           # inner-line charge on the bond after site 2
end
