using Test
using OpSum
using OpSum: SiteOperator, LocalOperator
using TensorKit
using TensorKit: insertrightunit
import TensorKitTensors.SpinOperators as SO
import TensorKitTensors.FermionOperators as FO
import TensorKitTensors.HubbardOperators as HO

nletters(x::SiteOperator) = length(collect(pairs(x)))
nletters(x::LocalOperator) = length(x.terms)

@testset "memoised" begin
    @test spin_basis(SU2Irrep) === spin_basis(SU2Irrep)
    @test spin_basis(SU2Irrep) == spin_basis(SU2Irrep) && hash(spin_basis(SU2Irrep)) == hash(spin_basis(SU2Irrep))
    @test hubbard_ops(U1Irrep, SU2Irrep) === hubbard_ops(U1Irrep, SU2Irrep)
end

@testset "complete: every operator on the site is in the span" begin
    for B in (
            spin_basis(U1Irrep; spin = 1), spin_basis(SU2Irrep; spin = 3 // 2), fermion_basis(),
            boson_basis(U1Irrep; cutoff = 3), hubbard_basis(U1Irrep, SU2Irrep), tj_basis(U1Irrep, SU2Irrep),
        )
        V = space(B)
        @test space(B[1]) == (V ← V ⊗ oneunit(V)) && B[1] ≈ insertrightunit(id(V))
        @test project(randn(ComplexF64, V ← V), B) isa SiteOperator
        @test project(randn(ComplexF64, V ⊗ V ← V ⊗ V), B) isa LocalOperator
    end
    # so is a hand-made basis: the given operators first, then matrix units
    Xm = ComplexF64[0 1; 1 0]
    B = Basis([Xm])
    @test length(B) == 4 && TensorKit.removeunit(B[2], 3) ≈ TensorMap(Xm, ℂ^2 ← ℂ^2)
    @test project(randn(ComplexF64, 2, 2), B) isa SiteOperator
end

@testset "the common operators are one letter" begin
    @test nletters(spin_ops(SU2Irrep).S_exchange) == 1
    @test nletters(spin_ops(U1Irrep).S_exchange) == 3            # Sᶻ Sᶻ and the two S± pairs
    @test nletters(spin_ops(U1Irrep).S_z) == 1
    @test nletters(spin_ops(Z2Irrep).S_z_S_z) == 1
    @test nletters(fermion_ops(Trivial).f_plus_f_min) == 1
    @test nletters(hubbard_ops(U1Irrep, SU2Irrep).e_plus_e_min) == 1
    @test nletters(hubbard_ops(U1Irrep, U1Irrep).e_plus_e_min) == 2    # one per spin
    # operators the symmetry does not admit as neutral ones are absent
    @test !haskey(spin_ops(SU2Irrep), :S_z)
    @test !haskey(spin_ops(U1Irrep), :S_plus)
end

@testset "the builders are TensorKitTensors' operators" begin
    for (ops, B, ref) in (
            (spin_ops(U1Irrep), spin_basis(U1Irrep), (; S_z = SO.S_z(ComplexF64, U1Irrep), S_exchange = SO.S_exchange(ComplexF64, U1Irrep))),
            (fermion_ops(Trivial), fermion_basis(), (; f_num = FO.f_num(ComplexF64, Trivial), f_hopping = FO.f_hopping(ComplexF64, Trivial))),
            (hubbard_ops(U1Irrep, SU2Irrep), hubbard_basis(U1Irrep, SU2Irrep), (; ud_num = HO.ud_num(ComplexF64, U1Irrep, SU2Irrep))),
        )
        for (name, t) in pairs(ref)
            op = getfield(ops, name)
            D = op isa SiteOperator ? instantiate(op, B) : instantiate(opsum(FiniteChain(B, 2), op[1, 2]))
            @test D ≈ insertrightunit(t)
        end
    end
end

@testset "acceptance: XXX on ℂ² in the stock basis" begin
    ops = spin_ops()
    N = 4
    H = opsum(FiniteChain(spin_basis(), N), (ops.S_exchange[i, i + 1] for i in 1:(N - 1)))
    @test islossless(H)
    @test sum(dim, mpo(H).bondsectors[N ÷ 2]) == 5   # 6 in matrix units
end
