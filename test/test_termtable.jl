using Test
using OpSum
using OpSum: TermTable, Trie, build_trie!, arity, nterms, nvertices, algebratype
using OpSum.PauliOperators: I, X, Y, Z, PauliOperator
using VectorInterface: scalartype

# Canonical sparse form of a dense trie opstring: sorted non-identity (site => op).
sparse_key(ops) = [i => ops[i] for i in eachindex(ops) if !isone(ops[i])]

# The reference term content, taken from the existing Trie-based `build_trie!`,
# as a Dict from canonical sparse key to summed coefficient.
function trie_terms(vertices, ex)
    T = scalartype(ex)
    A = algebratype(typeof(ex))
    trie = Trie{A, T}()
    build_trie!(trie, vertices, ex, one(T))
    d = Dict{Vector{Pair{Int, A}}, T}()
    for (ops, c) in pairs(trie)
        k = sparse_key(ops)
        d[k] = get(d, k, zero(T)) + c
    end
    return d
end

# The same content read out of a TermTable.
function termtable_terms(vertices, ex)
    tt = TermTable(vertices, ex)
    T = scalartype(ex)
    A = algebratype(typeof(ex))
    d = Dict{Vector{Pair{Int, A}}, T}()
    for (k, c) in tt
        @test !haskey(d, k)   # columns are canonical: no duplicate keys
        d[k] = c
    end
    return d, tt
end

# Compare two key => coeff dictionaries: identical key sets, coefficients ≈.
function dicts_match(ref, got)
    keys(ref) == keys(got) || return false
    return all(k -> ref[k] ≈ got[k], keys(ref))
end

function terms_match(vertices, ex)
    ref = trie_terms(vertices, ex)
    got, tt = termtable_terms(vertices, ex)
    return dicts_match(ref, got), ref, got, tt
end

@testset "TermTable" begin

    @testset "matches build_trie! content — hand-built Hamiltonians" begin
        L = 5
        vertices = 1:L
        hamiltonians = [
            2.0 * X[1] * X[2],
            sum(Z[i] for i in 1:L),
            sum(X[i] * X[i + 1] for i in 1:(L - 1)),
            sum(X[i] * X[i + 1] for i in 1:(L - 1)) + sum(Z[i] for i in 1:L),
            sum(X[i] * X[i + 1] + Y[i] * Y[i + 1] + Z[i] * Z[i + 1] for i in 1:(L - 1)),
            sum(X[i] * X[j] for i in 1:(L - 1) for j in (i + 1):L),
            1.5 * X[1] * X[3] + 0.7 * X[2] * X[3],
        ]
        for H in hamiltonians
            ok, ref, got, tt = terms_match(vertices, H)
            @test ok
            @test nvertices(tt) == L
            @test arity(tt) >= 2
        end
    end

    @testset "structural invariants" begin
        H = sum(X[i] * X[j] for i in 1:4 for j in (i + 1):5)
        tt = TermTable(1:5, H)
        # sites sorted ascending, zero-padded at the end
        for t in 1:nterms(tt)
            col = tt.sites[:, t]
            occupied = filter(!=(0), col)
            @test issorted(occupied)
            @test allunique(occupied)
            # zeros only appear after occupied entries
            firstzero = findfirst(==(0), col)
            if firstzero !== nothing
                @test all(==(0), col[firstzero:end])
            end
            # padded op slots are identity
            for j in 1:arity(tt)
                tt.sites[j, t] == 0 && @test isone(tt.ops[j, t])
            end
        end
        # two-body terms => arity exactly 2
        @test arity(tt) == 2
        @test nterms(tt) == 10   # C(5,2)
    end

    @testset "duplicate-term coefficient accumulation" begin
        # Two syntactically distinct GlobalOp terms that expand to the SAME
        # sparse operator string must merge with summed coefficients.
        H = 1.5 * X[1] * X[2] + 2.5 * X[1] * X[2]
        tt = TermTable(1:3, H)
        @test nterms(tt) == 1
        (k, c), _ = iterate(tt)
        @test c ≈ 4.0
        @test k == [1 => PauliOperator(0x01), 2 => PauliOperator(0x01)]

        # Cross-check against the Trie path.
        ok, = terms_match(1:3, H)
        @test ok
    end

    @testset "fuzz equivalence vs build_trie!" begin
        ops = [X, Y, Z]
        # deterministic LCG, so failures are reproducible without an RNG import
        seed = Ref(12345)
        rand01() = (seed[] = (seed[] * 1103515245 + 12345) % 2147483648; seed[] / 2147483648)
        randint(n) = 1 + Int(floor(rand01() * n))
        for trial in 1:40
            L = 2 + randint(5)                       # 3..7 sites
            nterm = randint(10)                      # 1..10 terms
            H = nothing
            for _ in 1:nterm
                k = min(randint(3), L)               # 1..3 factors
                sites = sort(collect(first(unique(randint(L) for _ in 1:50), k)))
                coeff = round(rand01() * 4 - 2; digits = 3)
                iszero(coeff) && (coeff = 1.0)
                term = coeff * prod(ops[randint(3)][s] for s in sites)
                H = H === nothing ? term : H + term
            end
            ok, ref, got = terms_match(1:L, H)
            @test ok
        end
    end

end
