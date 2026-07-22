using Test
using OpSum
using OpSum: instantiate, irrep_trie, trie_terms, TermSum, TermKey, ITOKey, total,
    passthrough, ispassthrough, bondcharges, vertexlabels, caterpillar_trees,
    spin, scalarop, couple
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot, I as Id

LO(x) = OpSum.LocalOp(x)

# the single (TermKey, coeff) of a one-term TermSum
onlyterm(ts::TermSum) = only(pairs(ts.terms))
# charges of a term's active operators
charges(tk::TermKey) = [o.c for o in tk.ops]
# per-active-position bond charges of a term (from its stored caterpillar tree)
termbonds(tk::TermKey) = bondcharges(tk.tree)

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
    @test length(ts.terms) == 1
    tk, coeff = onlyterm(ts)

    @test tk.sites == [1, 2]
    @test charges(tk) == [SU2Irrep(1), SU2Irrep(1)]
    @test total(tk) == unit(SU2Irrep)                   # singlet
    @test termbonds(tk) == [SU2Irrep(1), SU2Irrep(0)]   # bond 1 across the pair, 0 at the total
    @test coeff ≈ (3 / 2) * (-sqrt(3))                  # ⟨½‖S‖½⟩² · (−√dim(1))

    # non-singlet target: unit Clebsch–Gordan (no −√dim factor)
    ts2 = couple(spin(V)[1], spin(V)[2]; to = SU2Irrep(1))
    tk2, c2 = onlyterm(ts2)
    @test c2 ≈ (3 / 2)
    @test termbonds(tk2) == [SU2Irrep(1), SU2Irrep(1)]
end

@testset "caterpillar channel enumeration is bounded/canonical" begin
    I = SU2Irrep
    c = SU2Irrep(1)
    @test length(caterpillar_trees((c, c), SU2Irrep(0))) == 1
    @test length(caterpillar_trees((c, c, c), SU2Irrep(0))) == 1
    @test length(caterpillar_trees((c, c, c), SU2Irrep(1))) > 1   # 3-body: multiple channels (pick via nested couple)
end

@testset "SU(2) trie is charge-block-diagonal" begin
    V = SU2Space(1 // 2 => 1)

    # prefix sharing: S₁·S₂ and S₁·S₃ agree on the site-1 (op, bond) → shared node
    tr = irrep_trie([dot(spin(V)[1], spin(V)[2]), dot(spin(V)[1], spin(V)[3])], [V, V, V])
    @test length(tr.children) == 1
    site1_key = only(keys(tr.children))
    @test site1_key.op == IrrepOperator{SU2Irrep}(SU2Irrep(1), 1)
    @test site1_key.bond == SU2Irrep(1)
    @test length(only(tr.children).children) == 2      # diverge at site 2

    # distinct couplings on the same string → distinct paths (block-diagonal in bond)
    tr2 = irrep_trie(
        [
            couple(spin(V)[1], spin(V)[2]; to = SU2Irrep(0)),
            couple(spin(V)[1], spin(V)[2]; to = SU2Irrep(1)),
        ],
        [V, V],
    )
    @test length(tr2.children) == 1
    node = only(tr2.children)
    site2_bonds = sort!([k.bond.j for k in keys(node.children)])
    @test site2_bonds == [0 // 1, 1 // 1]
end

@testset "round-trip: trie ↔ term-sum" begin
    V = SU2Space(1 // 2 => 1)
    sites = [V, V, V, V]
    H = reduce(+, dot(spin(V)[i], spin(V)[i + 1]) for i in 1:3)
    back = trie_terms(irrep_trie(H, sites))

    @test Set(keys(back.terms)) == Set(keys(H.terms))
    for k in keys(H.terms)
        @test back.terms[k] ≈ H.terms[k]
    end
end

@testset "U(1) hopping term" begin
    V = Rep[U₁](0 => 1, 1 => 1)
    raise = LO(IrrepOperator(U1Irrep(1), 1))
    lower = LO(IrrepOperator(U1Irrep(-1), 1))

    ts = dot(raise[1], lower[2])
    tk, coeff = onlyterm(ts)
    @test charges(tk) == [U1Irrep(1), U1Irrep(-1)]
    @test total(tk) == unit(U1Irrep)
    @test termbonds(tk) == [U1Irrep(1), U1Irrep(0)]
    @test coeff ≈ -1.0                                  # dot: -√dim(1) · (1·1) = -1

    ts2 = couple(raise[1], raise[2]; to = U1Irrep(2))
    @test termbonds(onlyterm(ts2)[1]) == [U1Irrep(1), U1Irrep(2)]

    tr = irrep_trie([dot(raise[1], lower[2]), dot(raise[2], lower[3])], [V, V, V])
    @test length(trie_terms(tr).terms) == 2
end

@testset "trivial-sector bridge (ℂ²)" begin
    V = ℂ^2
    ops = instances(IrrepOperator, V)
    term = couple(LO(ops[2])[1], LO(ops[3])[2]; to = unit(Trivial))
    tk, coeff = onlyterm(term)
    @test termbonds(tk) == [unit(Trivial), unit(Trivial)]
    @test coeff ≈ 1.0                                   # bare `couple` carries no factor (dot would give -1)
    @test !ispassthrough(tk.ops[1]) && !ispassthrough(tk.ops[2])

    # a scalar (identity) field is an all-pass-through path (K = 0 term)
    idterm = scalarop(2.0, V)[1]
    tk0, c0 = onlyterm(idterm)
    @test isempty(tk0.sites)
    @test c0 ≈ 2.0
    path = only(keys(irrep_trie(idterm, [V, V]).children)).op
    @test ispassthrough(path)

    # round-trip on a trivial-sector chain
    back = trie_terms(irrep_trie(term, [V, V]))
    @test Set(keys(back.terms)) == Set(keys(term.terms))
end

# Closes the Phase-3 semantic caveat: the *reduced* representation (reduced coefficients + fusion
# structure), routed through the trie and back, must re-materialize to the correct dense operator
# — verified end-to-end against an explicit Heisenberg chain (multi-term, multi-site).
@testset "end-to-end: reduced representation reconstructs the dense operator" begin
    V = SU2Space(1 // 2 => 1)
    Sx = ComplexF64[0 1; 1 0] / 2
    Sy = ComplexF64[0 -im; im 0] / 2
    Sz = ComplexF64[1 0; 0 -1] / 2
    Svec = (Sx, Sy, Sz)
    I2 = Matrix{ComplexF64}(Id, 2, 2)

    # explicit dense Heisenberg chain  Σ_i Σ_a Sᵃ_i Sᵃ_{i+1}
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
        viatrie = instantiate(trie_terms(irrep_trie(H, sites)), sites)
        # trie round-trip re-materializes to the same operator
        @test convert(Array, viatrie) ≈ convert(Array, direct)
        # and that operator is the physical Heisenberg chain (end-to-end, multi-term)
        @test to_matrix(direct, N) ≈ heis_ref(N)
    end
end

# K ≥ 3: the intermediate ("inner line") charge is a genuine degree of freedom that `(charges,
# total)` cannot capture — the stored caterpillar tree keeps the channels distinct.
@testset "K=3 SU(2): distinct coupling channels" begin
    V = SU2Space(1 // 2 => 1)
    Sop = spin(V)
    channels = (SU2Irrep(0), SU2Irrep(1), SU2Irrep(2))     # (S₁⊗S₂) → b₂, all fuse with S₃ to a singlet? use total 1
    terms = [couple(couple(Sop[1], Sop[2]; to = b2), Sop[3]; to = SU2Irrep(1)) for b2 in channels]
    ks = [onlyterm(t)[1] for t in terms]

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

    # round-trip through the trie preserves the (tree-resolved) term identity for each channel
    for t in terms
        back = trie_terms(irrep_trie(t, fill(V, 3)))
        @test Set(keys(back.terms)) == Set(keys(t.terms))
    end
end
