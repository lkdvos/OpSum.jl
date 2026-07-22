using Test
using OpSum
using OpSum: mpo_to_dense, instantiate, TermTable, BipartiteAlgorithm, SVDBondAlgorithm,
    Trie, build_trie!, algebratype
using OpSum.PauliOperators: X, Y, Z
using VectorInterface: scalartype
using LinearAlgebra: norm

# Explicit Trie-based reference (the default now routes GlobalOp through TermTable).
function trie_optimize(vertices, ex, alg)
    T = scalartype(ex)
    A = algebratype(typeof(ex))
    trie = Trie{A, T}()
    build_trie!(trie, vertices, ex, one(T))
    return mpo_bond_optimizations(vertices, trie, alg)
end

# Reconstructed dense operator via the flat TermTable bond-optimization path.
function termtable_dense(vertices, ex, sites)
    tt = TermTable(vertices, ex)
    Ws = mpo_bond_optimizations(vertices, tt, BipartiteAlgorithm())
    return mpo_to_dense(Ws, sites), Ws
end

@testset "TermTable MPO bond optimization" begin

    @testset "matches dense operator + Trie path — standard Hamiltonians" begin
        L = 5
        vertices = 1:L
        sites = fill(2, L)
        hamiltonians = [
            2.0 * X[1] * X[2],
            sum(Z[i] for i in 1:L),
            sum(rand() * Z[i] for i in 1:L),
            sum(X[i] * X[i + 1] for i in 1:(L - 1)),
            sum(X[i] * X[i + 1] for i in 1:(L - 1)) + sum(Z[i] for i in 1:L),
            sum(X[i] * X[i + 1] + Y[i] * Y[i + 1] + Z[i] * Z[i + 1] for i in 1:(L - 1)),
            sum(X[i] * X[j] for i in 1:(L - 1) for j in (i + 1):L),
        ]
        for H in hamiltonians
            H_ref = instantiate(H, sites)
            # genuine Trie path (built explicitly; the GlobalOp default now uses TermTable)
            Ws_old = trie_optimize(vertices, H, BipartiteAlgorithm())
            @test mpo_to_dense(Ws_old, sites) ≈ H_ref
            # new (TermTable) path
            H_new, Ws_new = termtable_dense(vertices, H, sites)
            @test H_new ≈ H_ref
            @test length(Ws_new) == L
        end
    end

    @testset "mixed coefficients on 3 sites" begin
        H = 1.5 * X[1] * X[3] + 0.7 * X[2] * X[3]
        sites = fill(2, 3)
        H_new, Ws = termtable_dense(1:3, H, sites)
        @test H_new ≈ instantiate(H, sites)
        @test length(Ws) == 3
    end

    @testset "duplicate-term coefficient accumulation reconstructs correctly" begin
        H = 1.5 * X[1] * X[2] + 2.5 * X[1] * X[2]
        sites = fill(2, 3)
        H_new, = termtable_dense(1:3, H, sites)
        @test H_new ≈ instantiate(H, sites)
        @test H_new ≈ instantiate(4.0 * X[1] * X[2], sites)
    end

    @testset "bond dimension is competitive with the Trie path" begin
        # For an XX chain the min-vertex-cover bond dim should not exceed the
        # Trie path's (both solve the same per-bond cover).
        L = 6
        vertices = 1:L
        H = sum(X[i] * X[i + 1] for i in 1:(L - 1)) + sum(Z[i] for i in 1:L)
        Ws_old = mpo_bond_optimizations(vertices, H)
        tt = TermTable(vertices, H)
        Ws_new = mpo_bond_optimizations(vertices, tt, BipartiteAlgorithm())
        dmax_old = maximum(x -> max(size(x)...), Ws_old)
        dmax_new = maximum(x -> max(size(x)...), Ws_new)
        @test dmax_new <= dmax_old
    end

    @testset "SVD path: TermTable matches Trie SVD and dense operator" begin
        L = 5
        vertices = 1:L
        sites = fill(2, L)
        hamiltonians = [
            2.0 * X[1] * X[2],
            sum(X[i] * X[i + 1] for i in 1:(L - 1)),
            sum(X[i] * X[i + 1] + Y[i] * Y[i + 1] + Z[i] * Z[i + 1] for i in 1:(L - 1)),
            sum(X[i] * X[j] for i in 1:(L - 1) for j in (i + 1):L),
            1.5 * X[1] * X[3] + 0.7 * X[2] * X[3],
        ]
        for H in hamiltonians
            H_ref = instantiate(H, sites)
            Ws_trie = trie_optimize(vertices, H, SVDBondAlgorithm())
            tt = TermTable(vertices, H)
            Ws_tt = mpo_bond_optimizations(vertices, tt, SVDBondAlgorithm())
            @test mpo_to_dense(Ws_trie, sites) ≈ H_ref
            @test mpo_to_dense(Ws_tt, sites) ≈ H_ref
            # same bond dimensions as the Trie SVD (identical algorithm, flat source)
            @test size.(Ws_tt) == size.(Ws_trie)
        end
    end

    @testset "fuzz equivalence vs Trie path (dense operator)" begin
        ops = [X, Y, Z]
        seed = Ref(2024)
        rand01() = (seed[] = (seed[] * 1103515245 + 12345) % 2147483648; seed[] / 2147483648)
        randint(n) = 1 + Int(floor(rand01() * n))
        for trial in 1:30
            L = 2 + randint(4)                      # 3..6 sites
            sites = fill(2, L)
            nterm = randint(8)
            H = nothing
            for _ in 1:nterm
                k = min(randint(3), L)
                sts = sort(collect(first(unique(randint(L) for _ in 1:50), k)))
                coeff = round(rand01() * 4 - 2; digits = 3)
                iszero(coeff) && (coeff = 1.0)
                term = coeff * prod(ops[randint(3)][s] for s in sts)
                H = H === nothing ? term : H + term
            end
            H_ref = instantiate(H, sites)
            H_new, = termtable_dense(1:L, H, sites)
            @test H_new ≈ H_ref
        end
    end

end
