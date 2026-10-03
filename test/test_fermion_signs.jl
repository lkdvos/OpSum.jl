using Test
using OpSum
using OpSum: instantiate, couple, opsum, _isfermionodd
using TensorKit
using TensorKit: removeunit, numind
using LinearAlgebra

# `couple(cd[i], c[j])` is the physical `c†ᵢcⱼ` (Jordan–Wigner `c_j = Z₁⋯Z_{j-1} σ⁻_j`, sites 1..N);
# the hop sign is physical on odd loops. Pinned against a dense oracle written independently of OpSum.
const Vf = Vect[FermionNumber](0 => 1, 1 => 1)
const vac, occ = FermionNumber(0), FermionNumber(1)
const Zp = ComplexF64[1 0; 0 -1]
const I2 = ComplexF64[1 0; 0 1]
jw_c(j, N) = foldl(kron, [k < j ? Zp : (k == j ? ComplexF64[0 1; 0 0] : I2) for k in 1:N])
jw_cd(j, N) = Matrix(jw_c(j, N)')

# product-basis matrix of a fermionic TensorMap, read blockwise (site 1 most significant)
function densemat(t::AbstractTensorMap)
    numind(t) == 2numout(t) || (t = removeunit(t, numind(t)))
    N = numout(t)
    idx(ns) = 1 + sum(ns[k] << (N - k) for k in 1:N)
    M = zeros(ComplexF64, 2^N, 2^N)
    for (f1, f2) in fusiontrees(t)
        no = [Int(a.sectors[1].charge) for a in f1.uncoupled]
        ni = [Int(a.sectors[1].charge) for a in f2.uncoupled]
        M[idx(no), idx(ni)] = only(t[f1, f2])
    end
    return M
end
dense(h, N) = densemat(instantiate(h, FiniteChain(Vf, N)))

@testset "fermionic signs" begin
    F = fermion_ops(Vf)
    @testset "hops and four-fermion term" begin
        for N in 2:4, i in 1:N, j in 1:N
            i == j || @test dense(couple(F.cd[i], F.c[j]), N) ≈ jw_cd(i, N) * jw_c(j, N)
        end
        @test dense(couple(F.cd[1], F.cd[2], F.c[3], F.c[4]), 4) ≈
            jw_cd(1, 4) * jw_cd(2, 4) * jw_c(3, 4) * jw_c(4, 4)
        @test couple(F.c[2], F.cd[1]) ≈ -couple(F.cd[1], F.c[2])
    end

    @testset "triangle" begin
        bonds = [(1, 2), (2, 3), (1, 3)]
        hop(i, j) = -(couple(F.cd[i], F.c[j]) + couple(F.cd[j], F.c[i]))
        Hjw = sum(-(jw_cd(i, 3) * jw_c(j, 3) + jw_cd(j, 3) * jw_c(i, 3)) for (i, j) in bonds)
        h = opsum(hop(i, j) for (i, j) in bonds)
        @test dense(h, 3) ≈ Hjw
        one = [count_ones(m) == 1 for m in 0:7]
        @test sort(real(eigvals(Hermitian(Hjw[one, one])))) ≈ [-2.0, 1.0, 1.0]
        @test islossless(h, FiniteChain(Vf, 3))
    end

    @testset "project inverts instantiate" begin
        leg(a, b, c) = only(fusiontrees((a, b), c, (false, false)))
        t = zeros(ComplexF64, Vf ⊗ Vf ← Vf ⊗ Vf)
        t[leg(occ, vac, occ), leg(vac, occ, occ)] .= 1
        @test project(t, [1, 2]) ≈ couple(F.cd[1], F.c[2])
    end

    @testset "parity of product sectors" begin
        Hub = ProductSector{Tuple{U1Irrep, SU2Irrep, FermionParity}}
        @test _isfermionodd(Hub((1, 1 // 2, 1)))
        @test !_isfermionodd(Hub((2, 0, 0)))
        @test _isfermionodd(FermionNumber(1)) && !_isfermionodd(FermionNumber(2))
        @test !_isfermionodd(SU2Irrep(1 // 2)) && !_isfermionodd(U1Irrep(1))
    end
end
