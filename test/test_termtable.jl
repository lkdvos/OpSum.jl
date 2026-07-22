using Test
using OpSum
using OpSum: TermTable, arity, nterms, nvertices
using OpSum.PauliOperators: I, X, Y, Z, PauliOperator

# Underlying algebra elements (TermTable stores the bare operator, not the LocalOp wrapper).
const Px = PauliOperator(0x01)
const Py = PauliOperator(0x02)
const Pz = PauliOperator(0x03)

# TermTable content as a Dict from sparse key (sorted [site => op]) to coefficient.
content(tt) = Dict(collect(tt))

@testset "TermTable" begin

    @testset "explicit content" begin
        # single scaled two-body term
        tt = TermTable(1:3, 2.0 * X[1] * X[2])
        @test nterms(tt) == 1
        @test nvertices(tt) == 3
        @test arity(tt) == 2
        @test content(tt) == Dict([1 => Px, 2 => Px] => 2.0 + 0im)

        # sum of single-site fields
        tt = TermTable(1:4, sum(Z[i] for i in 1:4))
        @test nterms(tt) == 4
        @test content(tt) == Dict([i => Pz] => 1.0 + 0im for i in 1:4)

        # mixed coefficients / overlapping support
        tt = TermTable(1:3, 1.5 * X[1] * X[3] + 0.7 * X[2] * X[3])
        @test content(tt) == Dict(
            [1 => Px, 3 => Px] => 1.5 + 0im,
            [2 => Px, 3 => Px] => 0.7 + 0im,
        )
    end

    @testset "structural invariants" begin
        H = sum(X[i] * X[j] for i in 1:4 for j in (i + 1):5)
        tt = TermTable(1:5, H)
        @test arity(tt) == 2         # two-body terms
        @test nterms(tt) == 10       # C(5,2)
        for t in 1:nterms(tt)
            col = tt.sites[:, t]
            occupied = filter(!=(0), col)
            @test issorted(occupied)
            @test allunique(occupied)
            firstzero = findfirst(==(0), col)
            if firstzero !== nothing
                @test all(==(0), col[firstzero:end])   # zeros only trail
            end
            for j in 1:arity(tt)
                tt.sites[j, t] == 0 && @test isone(tt.ops[j, t])   # padded slots are identity
            end
        end
        # arity floor of 2 even when all terms are single-site
        @test arity(TermTable(1:4, sum(Z[i] for i in 1:4))) == 2
    end

    @testset "duplicate-term coefficient accumulation" begin
        # syntactically distinct terms expanding to the same sparse string merge
        tt = TermTable(1:3, 1.5 * X[1] * X[2] + 2.5 * X[1] * X[2])
        @test nterms(tt) == 1
        (k, c), _ = iterate(tt)
        @test k == [1 => Px, 2 => Px]
        @test c ≈ 4.0
    end

    @testset "iteration round-trips the stored columns" begin
        H = sum(X[i] * X[i + 1] for i in 1:4) + sum(Z[i] for i in 1:5)
        tt = TermTable(1:5, H)
        collected = collect(tt)
        @test length(collected) == nterms(tt)
        # every emitted key is sorted, non-empty, and identity-free
        for (k, _) in collected
            @test !isempty(k)
            @test issorted(k; by = first)
            @test all(p -> !isone(last(p)), k)
        end
    end

end
