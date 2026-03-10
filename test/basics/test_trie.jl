using Test
using Dictionaries: sortkeys, sortkeys!

using OpSum
using OpSum: Trie, subtrie, depth

# Helper: convert strings to Vector{Char} keys
chars(s) = collect(s)

@testset "Trie" begin

    @testset "Construction" begin
        t = Trie{Char, Int}()
        @test t isa Trie{Char, Int}
        @test isempty(t)

        # From keys and values
        ks = [chars("amy"), chars("ann"), chars("rob")]
        vs = [1, 2, 3]
        t2 = Trie(ks, vs)
        @test t2 isa Trie{Char, Int}
        @test length(t2) == 3

        t3 = Trie{Char, Int}(ks, vs)
        @test t3 isa Trie{Char, Int}
        @test length(t3) == 3
    end

    @testset "Type properties" begin
        t = Trie{Char, Int}()
        @test keytype(t) === Vector{Char}
        @test valtype(t) === Int
        @test eltype(t) === Int  # eltype is valtype for value iteration

        t2 = Trie{Int, Float64}()
        @test keytype(t2) === Vector{Int}
        @test valtype(t2) === Float64
    end

    @testset "setindex! and getindex" begin
        t = Trie{Char, Int}()
        t[chars("foo")] = 1
        @test t[chars("foo")] == 1

        # Overwrite existing key
        t[chars("foo")] = 99
        @test t[chars("foo")] == 99

        # Multiple keys
        t[chars("bar")] = 2
        t[chars("baz")] = 3
        @test t[chars("bar")] == 2
        @test t[chars("baz")] == 3

        # KeyError on missing key
        @test_throws KeyError t[chars("qux")]
        @test_throws KeyError t[chars("fo")]   # prefix of existing key, but not inserted
        @test_throws KeyError t[chars("foox")] # extension of existing key

        # Empty key (root value)
        t[Char[]] = 0
        @test t[Char[]] == 0
    end

    @testset "haskey" begin
        t = Trie{Char, Int}()
        t[chars("rob")] = 1
        t[chars("roger")] = 2
        t[chars("amy")] = 3

        @test haskey(t, chars("rob"))
        @test haskey(t, chars("roger"))
        @test haskey(t, chars("amy"))
        @test !haskey(t, chars("ro"))       # internal prefix without value
        @test !haskey(t, chars("rogers"))   # extension beyond existing
        @test !haskey(t, chars("am"))       # prefix of existing
        @test !haskey(t, chars(""))         # empty key not inserted
        @test !haskey(t, chars("xyz"))      # completely absent
    end

    @testset "isassigned" begin
        t = Trie{Char, Int}()
        t[chars("foo")] = 42
        @test Base.isassigned(t, chars("foo"))
        @test !Base.isassigned(t, chars("fo"))
        @test !Base.isassigned(t, chars("fooo"))
        @test !Base.isassigned(t, chars("bar"))
    end

    @testset "isempty" begin
        t = Trie{Char, Int}()
        @test isempty(t)
        t[chars("a")] = 1
        @test !isempty(t)
    end

    @testset "get with default" begin
        t = Trie{Char, Int}()
        t[chars("foo")] = 42

        @test get(t, chars("foo"), 0) == 42
        @test get(t, chars("bar"), 0) == 0
        @test get(t, chars("fo"), -1) == -1    # prefix not in trie
        @test get(t, chars("fooo"), -1) == -1  # extension not in trie
    end

    @testset "get with callable" begin
        t = Trie{Char, Int}()
        t[chars("foo")] = 42

        @test get(() -> -1, t, chars("foo")) == 42
        called = Ref(false)
        @test get(t, chars("bar")) do
            called[] = true
            99
        end == 99
        @test called[]
    end

    @testset "get!" begin
        t = Trie{Char, Int}()

        # Inserts default when key absent
        @test get!(t, chars("foo"), 7) == 7
        @test t[chars("foo")] == 7

        # Returns existing value when present
        @test get!(t, chars("foo"), 99) == 7
        @test t[chars("foo")] == 7

        # Callable form
        t2 = Trie{Char, Int}()
        called = Ref(false)
        @test get!(t2, chars("bar")) do
            called[] = true
            42
        end == 42
        @test called[]
        called[] = false
        @test get!(t2, chars("bar")) do
            called[] = true
            99
        end == 42
        @test !called[]  # callable not invoked when key exists
    end

    @testset "equality" begin
        t1 = Trie{Char, Int}()
        t2 = Trie{Char, Int}()
        @test t1 == t2
        @test isequal(t1, t2)

        t1[chars("foo")] = 1
        @test t1 != t2

        t2[chars("foo")] = 1
        @test t1 == t2
        @test isequal(t1, t2)

        t1[chars("bar")] = 2
        t2[chars("bar")] = 3
        @test t1 != t2
        @test !isequal(t1, t2)
    end

    @testset "length" begin
        t = Trie{Char, Int}()
        @test length(t) == 0

        t[chars("a")] = 1
        @test length(t) == 1

        t[chars("ab")] = 2
        t[chars("abc")] = 3
        @test length(t) == 3

        # Overwrite doesn't change length
        t[chars("a")] = 99
        @test length(t) == 3
    end

    @testset "subtrie" begin
        t = Trie{Char, Int}()
        t[chars("rob")] = 1
        t[chars("roger")] = 2
        t[chars("amy")] = 3

        # Existing prefix
        st = subtrie(t, chars("ro"))
        @test st isa Trie{Char, Int}
        @test !isnothing(st)
        @test isnothing(st.value)  # "ro" has no value

        st2 = subtrie(t, chars("rob"))
        @test !isnothing(st2)
        @test st2.value == 1

        # Non-existing prefix
        @test isnothing(subtrie(t, chars("xyz")))
        @test isnothing(subtrie(t, chars("rox")))

        # Empty prefix returns root
        @test subtrie(t, Char[]) === t
    end

    @testset "depth" begin
        t = Trie{Char, Int}()
        @test depth(t) == 0

        t[chars("a")] = 1
        @test depth(t) == 1

        t[chars("abc")] = 2
        @test depth(t) == 3

        t[chars("abcde")] = 3
        @test depth(t) == 5
    end

    @testset "Values iteration" begin
        t = Trie{Char, Int}()
        @test isempty(collect(t))

        t[chars("amy")] = 56
        t[chars("ann")] = 15
        t[chars("emma")] = 30
        t[chars("rob")] = 27
        t[chars("roger")] = 52
        t[chars("kevin")] = 11

        @test eltype(t) === Int
        @test sort(collect(t)) == [11, 15, 27, 30, 52, 56]
        @test length(collect(t)) == 6
    end

    @testset "Keys iteration" begin
        t = Trie{Char, Int}()
        ki = keys(t)
        @test eltype(ki) === Vector{Char}
        @test isempty(collect(ki))

        t[chars("amy")] = 1
        t[chars("ann")] = 2
        t[chars("emma")] = 3
        t[chars("rob")] = 4
        t[chars("roger")] = 5
        t[chars("kevin")] = 6

        ks = sort(collect(keys(t)))
        expected = sort(
            [
                chars("amy"), chars("ann"), chars("emma"),
                chars("rob"), chars("roger"), chars("kevin"),
            ]
        )
        @test ks == expected
        @test length(ks) == 6

        # keys returns independent copies (not aliased to internal state)
        collected = collect(keys(t))
        push!(collected[1], 'X')
        @test collected[1] != chars("amy")  # mutating copy doesn't affect others
        @test sort(collect(keys(t))) == expected  # trie unchanged
    end

    @testset "Pairs iteration" begin
        t = Trie{Char, Int}()
        pi = pairs(t)
        @test eltype(pi) === Pair{Vector{Char}, Int}
        @test isempty(collect(pi))

        t[chars("foo")] = 1
        t[chars("bar")] = 2
        t[chars("baz")] = 3

        ps = sort(collect(pairs(t)), by = first)
        @test ps == [chars("bar") => 2, chars("baz") => 3, chars("foo") => 1]

        # Values match getindex
        for (k, v) in pairs(t)
            @test t[k] == v
        end
    end

    @testset "Prefix key relationships" begin
        # This exercises the iterator bug-fix: internal nodes with both value and children
        t = Trie{Char, Int}()
        t[chars("ro")] = 1
        t[chars("rob")] = 2
        t[chars("roger")] = 3

        @test length(t) == 3
        @test haskey(t, chars("ro"))
        @test haskey(t, chars("rob"))
        @test haskey(t, chars("roger"))

        ks = sort(collect(keys(t)))
        @test ks == [chars("ro"), chars("rob"), chars("roger")]

        ps = Dict(k => v for (k, v) in pairs(t))
        @test ps[chars("ro")] == 1
        @test ps[chars("rob")] == 2
        @test ps[chars("roger")] == 3

        # Single-character prefix at root level
        t2 = Trie{Int, Int}()
        t2[[1]] = 10
        t2[[1, 2]] = 20
        t2[[1, 2, 3]] = 30
        @test length(t2) == 3
        @test sort(collect(t2)) == [10, 20, 30]

        # Empty key (root has value) with children
        t3 = Trie{Char, Int}()
        t3[Char[]] = 0
        t3[chars("a")] = 1
        t3[chars("ab")] = 2
        @test length(t3) == 3
        @test sort(collect(t3)) == [0, 1, 2]
        @test haskey(t3, Char[])
    end

    @testset "sortkeys" begin
        t = Trie{Int, Int}()
        t[[2, 1]] = 10
        t[[1, 3]] = 20
        t[[1, 2]] = 30

        ts = sortkeys(t)
        ks = collect(keys(ts))
        @test ks == [[1, 2], [1, 3], [2, 1]]

        # sortkeys! mutates in place
        sortkeys!(t)
        @test collect(keys(t)) == [[1, 2], [1, 3], [2, 1]]
    end

    @testset "show" begin
        showstr(t) = repr(MIME("text/plain"), t)

        # Empty trie
        t = Trie{Char, Int}()
        @test showstr(t) == "Trie{Char, Int64} (0 entries)"

        # Single entry — full chain compressed into one edge, singular "entry"
        t[chars("hello")] = 42
        @test showstr(t) == """
            Trie{Char, Int64} (1 entry)
            └─ "hello" => 42"""

        # Two entries with shared prefix: split visible, trailing chains compressed
        t2 = Trie{Char, Int}()
        t2[chars("rob")]   = 27
        t2[chars("roger")] = 52
        @test showstr(t2) == """
            Trie{Char, Int64} (2 entries)
            └─ "ro"
               ├─ "b" => 27
               └─ "ger" => 52"""

        # Internal node with value (prefix key relationship)
        t3 = Trie{Char, Int}()
        t3[chars("ro")]    = 1
        t3[chars("rob")]   = 2
        t3[chars("roger")] = 3
        @test showstr(t3) == """
            Trie{Char, Int64} (3 entries)
            └─ "ro" => 1
               ├─ "b" => 2
               └─ "ger" => 3"""

        # Root value (empty key) with children
        t4 = Trie{Char, Int}()
        t4[Char[]]    = 0
        t4[chars("a")] = 1
        @test showstr(t4) == """
            Trie{Char, Int64} (2 entries)
            ├─ [] => 0
            └─ "a" => 1"""

        # Multiple branches at root level — correct ├─ / └─ and │ continuation
        t5 = Trie{Char, Int}()
        t5[chars("amy")] = 1
        t5[chars("bob")] = 2
        t5[chars("cat")] = 3
        out = showstr(t5)
        @test startswith(out, "Trie{Char, Int64} (3 entries)")
        @test contains(out, "├─")  # at least one non-last branch
        @test contains(out, "└─")  # last branch
        @test contains(out, "=> 1")
        @test contains(out, "=> 2")
        @test contains(out, "=> 3")

        # Int keys: multi-element compressed edge shown as vector
        t6 = Trie{Int, String}()
        t6[[1, 2, 3]] = "abc"
        t6[[1, 2, 4]] = "abd"
        @test showstr(t6) == """
            Trie{Int64, String} (2 entries)
            └─ [1, 2]
               ├─ 3 => "abc"
               └─ 4 => "abd\""""
    end

    @testset "Integer key trie" begin
        t = Trie{Int, String}()
        t[[1, 2, 3]] = "abc"
        t[[1, 2, 4]] = "abd"
        t[[2, 0]] = "ba"

        @test t[[1, 2, 3]] == "abc"
        @test t[[1, 2, 4]] == "abd"
        @test t[[2, 0]] == "ba"
        @test length(t) == 3
        @test !haskey(t, [1, 2])
        @test haskey(t, [1, 2, 3])
    end

end
