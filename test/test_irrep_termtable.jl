using Test
using OpSum
using OpSum: irrep_mpo, irrep_mpo_flat, mpo_terms, ITOTermTable, arity, nterms, nvertices,
    TermSum, spin, scalarop, couple, storedpairs, _local_terms
using OpSum.IrrepTensorOperators: IrrepOperator
using TensorKit
using LinearAlgebra: dot

LO(x) = OpSum.LocalOp(x)

# Robust multiset of reduced entries: (site, l, r, letter charge, letter index, coeff). Avoids
# relying on LocalOp `==`/hash (which fall back to identity for the sum type).
function reduced_entries(Ws)
    out = Set{Any}()
    for (i, W) in enumerate(Ws)
        for (idx, lop) in storedpairs(W)
            l, r = Tuple(idx)
            for (letter, c) in _local_terms(lop)
                letter === nothing && continue
                push!(out, (i, l, r, letter.c, letter.n, round(c; digits = 10)))
            end
        end
    end
    return out
end

# lossless: the flat-storage MPO reconstructs the original term-sum exactly
function flat_lossless(H::TermSum, sites)
    Ws, secs = irrep_mpo_flat(H, sites)
    back = mpo_terms(Ws, secs)
    Set(keys(back.terms)) == Set(keys(H.terms)) || return false
    return all(back.terms[k] ≈ H.terms[k] for k in keys(H.terms))
end

# flat path matches the trie path bond-for-bond (sizes, bond sectors, and stored entries)
function flat_matches_trie(H::TermSum, sites)
    Wf, sf = irrep_mpo_flat(H, sites)
    Wt, st = irrep_mpo(H, sites)
    sf == st || return false
    size.(Wf) == size.(Wt) || return false
    return reduced_entries(Wf) == reduced_entries(Wt)
end

su2 = SU2Space(1 // 2 => 1)
u1 = Rep[U₁](0 => 1, 1 => 1)

@testset "ITOTermTable — flat storage + bipartite sweep" begin

    @testset "construction sanity" begin
        H = reduce(+, dot(spin(su2)[i], spin(su2)[i + 1]) for i in 1:3)
        tt = ITOTermTable(H, fill(su2, 4))
        @test nvertices(tt) == 4
        @test nterms(tt) == length(H.terms)
        # two-body terms => arity 2, sites sorted/zero-padded
        @test arity(tt) == 2
        for t in 1:nterms(tt)
            occ = filter(!=(0), tt.sites[:, t])
            @test issorted(occ) && allunique(occ)
        end
    end

    @testset "lossless — SU(2) Heisenberg" begin
        for N in (3, 4, 5)
            H = reduce(+, dot(spin(su2)[i], spin(su2)[i + 1]) for i in 1:(N - 1))
            @test flat_lossless(H, fill(su2, N))
        end
    end

    @testset "lossless — U(1) hopping" begin
        raise = LO(IrrepOperator(U1Irrep(1), 1))
        lower = LO(IrrepOperator(U1Irrep(-1), 1))
        N = 4
        H = reduce(+, dot(raise[i], lower[i + 1]) for i in 1:(N - 1)) +
            reduce(+, dot(lower[i], raise[i + 1]) for i in 1:(N - 1))
        @test flat_lossless(H, fill(u1, N))
    end

    @testset "lossless — K=3 three-body" begin
        S = spin(su2)
        H = couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))
        @test flat_lossless(H, fill(su2, 3))
    end

    @testset "matches trie path bond-for-bond" begin
        cases = [
            (reduce(+, dot(spin(su2)[i], spin(su2)[i + 1]) for i in 1:2), fill(su2, 3)),
            (reduce(+, dot(spin(su2)[i], spin(su2)[i + 1]) for i in 1:4), fill(su2, 5)),
            (reduce(+, dot(spin(su2)[i], spin(su2)[i + 1]) for i in 1:5), fill(su2, 6)),
            (spin(su2)[1] + spin(su2)[2] + spin(su2)[3], fill(su2, 3)),
            (scalarop(2.5, su2)[1] + dot(spin(su2)[1], spin(su2)[2]), fill(su2, 3)),
        ]
        for (H, sites) in cases
            @test flat_matches_trie(H, sites)
        end

        raise = LO(IrrepOperator(U1Irrep(1), 1))
        lower = LO(IrrepOperator(U1Irrep(-1), 1))
        N = 4
        Hu1 = reduce(+, dot(raise[i], lower[i + 1]) for i in 1:(N - 1)) +
            reduce(+, dot(lower[i], raise[i + 1]) for i in 1:(N - 1))
        @test flat_matches_trie(Hu1, fill(u1, N))
    end

end
