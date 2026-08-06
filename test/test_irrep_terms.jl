using Test
using OpSum
using OpSum: instantiate, Term, Terms, TermSum, total, passthrough, ispassthrough,
    bondcharges, caterpillar_trees, spin, scalarop, couple, opsum, tree
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot, I as Id

include(joinpath(@__DIR__, "testutils.jl"))   # LO, onlyterm

# charges of a term's active operators
charges(t::Term) = [k.op.c for k in t.keys]
# per-position bond charges, off the keys and — equivalently, the point — off the tree they rebuild
termbonds(t::Term) = [k.bond for k in t.keys]
treebonds(t::Term) = bondcharges(tree(t))

@testset "pass-through identity symbol" begin
    I = SU2Irrep
    pt = passthrough(I)
    @test ispassthrough(pt)
    @test pt.c == unit(I)
    @test pt.n == 0
    V = SU2Space(1 // 2 => 1)
    @test all(!ispassthrough, instances(IrrepOperator, V))
    @test convert(Array, instantiate(pt, V)) ≈ Matrix{ComplexF64}(Id, 2, 2)
end

@testset "SU(2) Sᵢ·Sⱼ term normal form" begin
    V = SU2Space(1 // 2 => 1)
    ts = dot(spin(V)[1], spin(V)[2])
    @test length(ts) == 1
    tk = onlyterm(ts)

    @test tk.sites == [1, 2]
    @test charges(tk) == [SU2Irrep(1), SU2Irrep(1)]
    @test total(tk) == unit(SU2Irrep)                   # singlet
    @test termbonds(tk) == [SU2Irrep(1), SU2Irrep(0)]   # bond 1 across the pair, 0 at the total
    @test treebonds(tk) == termbonds(tk)                # the keys *are* the tree
    @test tk.coeff ≈ (3 / 2) * (-sqrt(3))               # ⟨½‖S‖½⟩² · (−√dim(1))

    # non-singlet target: unit Clebsch–Gordan (no −√dim factor)
    tk2 = onlyterm(couple(spin(V)[1], spin(V)[2]; to = SU2Irrep(1)))
    @test tk2.coeff ≈ (3 / 2)
    @test termbonds(tk2) == [SU2Irrep(1), SU2Irrep(1)]
    @test treebonds(tk2) == termbonds(tk2)
end

@testset "caterpillar channel enumeration is bounded/canonical" begin
    I = SU2Irrep
    c = SU2Irrep(1)
    @test length(caterpillar_trees((c, c), SU2Irrep(0))) == 1
    @test length(caterpillar_trees((c, c, c), SU2Irrep(0))) == 1
    @test length(caterpillar_trees((c, c, c), SU2Irrep(1))) > 1   # 3-body: multiple channels
end

@testset "U(1) hopping term normal form" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    raise = LO(IrrepOperator(U1Irrep(1), 1))
    lower = LO(IrrepOperator(U1Irrep(-1), 1))

    tk = onlyterm(dot(raise[1], lower[2]))
    @test charges(tk) == [U1Irrep(1), U1Irrep(-1)]
    @test total(tk) == unit(U1Irrep)
    @test termbonds(tk) == [U1Irrep(1), U1Irrep(0)]
    @test treebonds(tk) == termbonds(tk)
    @test tk.coeff ≈ -1.0                               # dot: -√dim(1) · (1·1) = -1

    @test termbonds(onlyterm(couple(raise[1], raise[2]; to = U1Irrep(2)))) ==
        [U1Irrep(1), U1Irrep(2)]
end

# `dot` sorts before reading `-√dim(c)` off the left operand, so the factor cannot depend on the
# order written. Dual sectors have equal quantum dimension, so the two readings agree numerically too.
@testset "dot is order-independent in its operands" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    raise = LO(IrrepOperator(U1Irrep(1), 1))
    lower = LO(IrrepOperator(U1Irrep(-1), 1))
    @test dot(raise[1], lower[2]) ≈ dot(lower[2], raise[1])   # charges +1 and -1, not equal

    Vs = SU2Space(1 // 2 => 1)
    S = spin(Vs)
    @test dot(S[1], S[2]) ≈ dot(S[2], S[1])

    # charges that cannot reach the unit sector have no scalar product at all
    @test_throws ArgumentError dot(raise[1], raise[2])
    @test_throws ArgumentError dot(raise[1], raise[1])        # and not on the same site
end

@testset "trivial-sector bridge (ℂ²)" begin
    V = ℂ^2
    letters = instances(IrrepOperator, V)
    tk = onlyterm(couple(LO(letters[2])[1], LO(letters[3])[2]; to = unit(Trivial)))
    @test termbonds(tk) == [unit(Trivial), unit(Trivial)]
    @test tk.coeff ≈ 1.0                                # bare `couple` carries no factor
    @test !ispassthrough(tk.keys[1].op) && !ispassthrough(tk.keys[2].op)

    # a scalar (identity) field is a K = 0 term
    tk0 = onlyterm(scalarop(2.0, V)[1])
    @test isempty(tk0.sites)
    @test tk0.coeff ≈ 2.0
    @test total(tk0) == unit(Trivial)
end

# The reduced representation (reduced coefficients + fusion structure) must re-materialize to the
# correct dense operator — verified end-to-end against an explicit Heisenberg chain.
@testset "end-to-end: reduced representation reconstructs the dense operator" begin
    V = SU2Space(1 // 2 => 1)
    Sx = ComplexF64[0 1; 1 0] / 2
    Sy = ComplexF64[0 -im; im 0] / 2
    Sz = ComplexF64[1 0; 0 -1] / 2
    Svec = (Sx, Sy, Sz)
    I2 = Matrix{ComplexF64}(Id, 2, 2)

    function heis_ref(N)
        H = zeros(ComplexF64, 2^N, 2^N)
        for i in 1:(N - 1), Sa in Svec
            ops = [(k == i || k == i + 1) ? Sa : I2 for k in 1:N]
            H += foldl(kron, ops)
        end
        return H
    end
    # TensorMap operator → matrix: drop the dim-1 total-charge leg, reorder to kron index order
    function to_matrix(t, N)
        A = convert(Array, t)
        ndims(A) == 2N + 1 && (A = dropdims(A; dims = 2N + 1))
        perm = (reverse(1:N)..., reverse((N + 1):(2N))...)
        return reshape(permutedims(A, perm), 2^N, 2^N)
    end

    for N in (2, 3)
        sites = fill(V, N)
        H = reduce(+, dot(spin(V)[i], spin(V)[i + 1]) for i in 1:(N - 1))
        direct = instantiate(H, sites)
        @test to_matrix(direct, N) ≈ heis_ref(N)
    end
end

# K ≥ 3: the intermediate ("inner line") charge is a genuine degree of freedom that `(charges,
# total)` cannot capture — the stored caterpillar tree keeps the channels distinct.
@testset "K=3 SU(2): distinct coupling channels" begin
    V = SU2Space(1 // 2 => 1)
    Sop = spin(V)
    channels = (SU2Irrep(0), SU2Irrep(1), SU2Irrep(2))     # (S₁⊗S₂) → b₂, fuse with S₃ to total 1
    terms = [couple(couple(Sop[1], Sop[2]; to = b2), Sop[3]; to = SU2Irrep(1)) for b2 in channels]
    ks = [onlyterm(t) for t in terms]

    # same sites, ops, and total — distinguished ONLY by the inner-line charge in the tree
    @test length(unique(ks)) == 3
    @test all(k -> k.sites == [1, 2, 3] && total(k) == SU2Irrep(1), ks)
    @test [termbonds(k) for k in ks] == [
        [SU2Irrep(1), SU2Irrep(0), SU2Irrep(1)],
        [SU2Irrep(1), SU2Irrep(1), SU2Irrep(1)],
        [SU2Irrep(1), SU2Irrep(2), SU2Irrep(1)],
    ]

    # and they materialize to genuinely different operators
    mats = [convert(Array, instantiate(t, fill(V, 3))) for t in terms]
    @test !(mats[1] ≈ mats[2])
    @test !(mats[1] ≈ mats[3])
    @test !(mats[2] ≈ mats[3])
end
