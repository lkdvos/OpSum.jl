using Test
using OpSum
using OpSum: mpo_to_dense, instantiate, TermTable, BipartiteAlgorithm, SVDBondAlgorithm
using OpSum.PauliOperators: X, Y, Z
using LinearAlgebra: norm

# Reconstructed dense operator via the flat TermTable bond-optimization path.
function termtable_dense(vertices, ex, sites, alg = BipartiteAlgorithm())
    tt = TermTable(vertices, ex)
    Ws = mpo_bond_optimizations(vertices, tt, alg)
    return mpo_to_dense(Ws, sites), Ws
end

@testset "TermTable MPO bond optimization" begin

    @testset "reconstructs the dense operator — standard Hamiltonians" begin
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
            H_new, Ws = termtable_dense(vertices, H, sites)
            @test H_new ≈ H_ref
            @test length(Ws) == L
            # the GlobalOp entry (which now builds a TermTable internally) agrees
            @test mpo_to_dense(mpo_bond_optimizations(vertices, H), sites) ≈ H_ref
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

    @testset "SVD path reconstructs the dense operator" begin
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
            H_new, Ws = termtable_dense(vertices, H, sites, SVDBondAlgorithm())
            @test H_new ≈ instantiate(H, sites)
            @test length(Ws) == L
        end
    end

    @testset "fuzz: reconstructs the dense operator" begin
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
