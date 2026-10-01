using Test
using OpSum
using OpSum: LocalOperator, nsites, instantiate, couple, couple_channels, spin, spin_ops,
    fermion_ops, scalarop, opsum, irrep_mpo, irrep_mpo_tensors, mpo_tensormap, islossless,
    FiniteChain, Terms, Term, ispassthrough, total, arity
using TensorKit
using TensorKit: removeunit, numind
using LinearAlgebra: dot

include(joinpath(@__DIR__, "testutils.jl"))   # LO, onlyterm, physmatrix

# `instantiate` output, as the plain `V ← V` map (the trailing trivial charge leg dropped)
function densop(h, lat)
    O = instantiate(opsum(lat, h))
    return numind(O) == 2 * numout(O) ? O : removeunit(O, numind(O))
end

# the message of the error `f()` throws, for comparing placed and unplaced error paths
function errmsg(f)
    try
        f()
        return nothing
    catch e
        return sprint(showerror, e)
    end
end

const Vsu2 = SU2Space(1 // 2 => 1)
const Vu1 = Rep[U₁](0 => 1, 1 => 1)
const Vf = Vect[FermionNumber](0 => 1, 1 => 1)

const zero_, one_, two_ = SU2Irrep(0), SU2Irrep(1), SU2Irrep(2)

# placements of a K-slot block: contiguous and gapped, starting at 1 and later
placements(K) = (collect(1:K), collect(3:(K + 2)), collect(1:2:(2K - 1)), [2; collect(4:(K + 2))])

# The unplaced coupling is the placed one with the sites left for later: for every placement the
# term bags must be *structurally* equal (same keys, same coefficients), not just `≈`.
@testset "couple(a, b)[sites...] is couple(a[i], b[j])" begin
    @testset "SU(2)" begin
        S = spin(Vsu2)

        B = couple(S, S)
        @test B isa LocalOperator{SU2Irrep}
        @test nsites(B) == 2
        for (i, j) in ((1, 2), (3, 4), (1, 3), (2, 7))
            @test B[i, j] == couple(S[i], S[j])
            @test B[i, j] ≈ couple(S[i], S[j])
        end
        @test B[5] == couple(S[5], S[6])
        for c in (one_, two_)
            @test couple(S, S; to = c)[1, 3] == couple(S[1], S[3]; to = c)
            @test total(onlyterm(couple(S, S; to = c)[1])) == c
        end

        # three operands: forced channel, variadic and nested
        T3 = couple(S, S, S)
        @test nsites(T3) == 3
        N3 = couple(couple(S, S; to = one_), S)
        @test T3 == N3
        for sites in placements(3)
            @test T3[sites...] == couple(S[sites[1]], S[sites[2]], S[sites[3]])
            @test T3[sites...] ≈ couple(S[sites[1]], S[sites[2]], S[sites[3]])
            @test N3[sites...] == couple(couple(S[sites[1]], S[sites[2]]; to = one_), S[sites[3]])
        end
        # a charged total that is still forced (three spin-1 reach j = 3 only through j₁₂ = 2)
        @test couple(S, S, S; to = SU2Irrep(3))[1] == couple(S[1], S[2], S[3]; to = SU2Irrep(3))

        # four operands: every channel the query lists, nested on both sides
        for (j12, j123) in couple_channels(S, S, S, S)
            T4 = couple(couple(couple(S, S; to = j12), S; to = j123), S)
            @test nsites(T4) == 4
            for sites in placements(4)
                ref = couple(
                    couple(couple(S[sites[1]], S[sites[2]]; to = j12), S[sites[3]]; to = j123),
                    S[sites[4]],
                )
                @test T4[sites...] == ref
                @test T4[sites...] ≈ ref
            end
        end
        # a forced four-fold: (S⊗S)₂ ⊗ S → 1 → S, which fixes j₁₂₃ = 1
        T4f = couple(couple(S, S; to = two_), S, S)
        @test T4f[1, 2, 3, 4] == couple(couple(S[1], S[2]; to = two_), S[3], S[4])
    end

    @testset "U(1)" begin
        (; Sp, Sm, Sz) = spin_ops(Vu1, U1Irrep(1), U1Irrep(0))

        for (a, b, c) in ((Sp, Sm, unit(U1Irrep)), (Sz, Sz, unit(U1Irrep)), (Sp, Sp, U1Irrep(2)))
            B = couple(a, b; to = c)
            @test nsites(B) == 2
            for sites in placements(2)
                @test B[sites...] == couple(a[sites[1]], b[sites[2]]; to = c)
                @test B[sites...] ≈ couple(a[sites[1]], b[sites[2]]; to = c)
            end
        end
        # composite operands distribute, unplaced as placed
        @test length(couple(Sz, Sz)) == 4
        @test length(couple(Sz, Sz, Sz)) == 8
        @test couple(Sz, Sz, Sz)[1, 3, 5] == couple(Sz[1], Sz[3], Sz[5])

        # variadic, with every intermediate forced
        T = couple(Sp, Sm, Sz)
        for sites in placements(3)
            @test T[sites...] == couple(Sp[sites[1]], Sm[sites[2]], Sz[sites[3]])
        end
        @test couple(Sp, Sp, Sm; to = U1Irrep(1))[2] == couple(Sp[2], Sp[3], Sm[4]; to = U1Irrep(1))
    end

    @testset "fermions" begin
        F = fermion_ops(Vf)
        u = unit(FermionNumber)

        for (a, b) in ((F.cd, F.c), (F.c, F.cd), (F.n, F.n))
            B = couple(a, b)
            for sites in placements(2)
                @test B[sites...] == couple(a[sites[1]], b[sites[2]])
                @test B[sites...] ≈ couple(a[sites[1]], b[sites[2]])
            end
        end
        # the unplaced path never reorders, so `couple(c, cd)` is the site-ordered `c_i c†_j`,
        # and the out-of-order placed spelling differs from it by the anticommutation sign
        @test couple(F.c, F.cd)[1, 2] ≈ -couple(F.cd[2], F.c[1])
        @test couple(F.cd, F.c; to = u)[1, 4] == couple(F.cd[1], F.c[4])

        # four-fermion variadic, contiguous and across gaps
        H4 = couple(F.cd, F.c, F.cd, F.c)
        @test nsites(H4) == 4
        for sites in placements(4)
            ref = couple(F.cd[sites[1]], F.c[sites[2]], F.cd[sites[3]], F.c[sites[4]])
            @test H4[sites...] == ref
            @test H4[sites...] ≈ ref
        end
        # and the nested form with the channels named
        nested = couple(couple(couple(F.cd, F.c; to = u), F.cd; to = FermionNumber(1)), F.c; to = u)
        @test nested == H4
    end
end

@testset "dot(a, b)[i, j] is dot(a[i], b[j])" begin
    S = spin(Vsu2)
    b = dot(S, S)
    @test b isa LocalOperator{SU2Irrep}
    @test nsites(b) == 2
    for sites in placements(2)
        @test b[sites...] == dot(S[sites[1]], S[sites[2]])
        @test b[sites...] ≈ dot(S[sites[1]], S[sites[2]])
    end
    @test b[4] == dot(S[4], S[5])
    # the Cartesian factor is the placed one
    @test onlyterm(b[1]).coeff ≈ -sqrt(3) * onlyterm(couple(S[1], S[2])).coeff

    # abelian and fermionic scalar products, where the letter pairs are duals
    (; Sp, Sm) = spin_ops(Vu1, U1Irrep(1), U1Irrep(0))
    @test dot(Sp, Sm)[1, 3] == dot(Sp[1], Sm[3])
    @test dot(Sm, Sp)[2, 3] == dot(Sm[2], Sp[3])
    F = fermion_ops(Vf)
    @test dot(F.cd, F.c)[1, 2] == dot(F.cd[1], F.c[2])
    @test dot(F.c, F.cd)[1, 4] == dot(F.c[1], F.cd[4])
    # never reversed: the slots are in argument order, so the swap phase of the placed
    # out-of-order `dot` does not enter here
    @test dot(F.c, F.cd)[1, 2] ≈ -dot(F.cd[2], F.c[1])
    @test dot(S, S)[1, 2] ≈ dot(S[2], S[1])     # integer charge: R = +1
end

@testset "couple_channels: unplaced == placed" begin
    S = spin(Vsu2)
    @test couple_channels(S, S) == couple_channels(S[1], S[2]) == [()]
    @test isempty(couple_channels(S, S; to = SU2Irrep(5)))
    @test couple_channels(S, S, S) == couple_channels(S[1], S[2], S[3]) == [(one_,)]
    @test couple_channels(S, S, S; to = one_) == couple_channels(S[1], S[2], S[3]; to = one_)
    @test couple_channels(S, S, S, S) == couple_channels(S[1], S[2], S[3], S[4])
    @test couple_channels(S, S, S, S) == [(zero_, one_), (one_, one_), (two_, one_)]
    # a multi-slot first operand contributes its running totals
    @test couple_channels(couple(S, S; to = one_), S, S) ==
        couple_channels(couple(S[1], S[2]; to = one_), S[3], S[4])

    F = fermion_ops(Vf)
    @test couple_channels(F.cd, F.c, F.cd, F.c) == couple_channels(F.cd[1], F.c[2], F.cd[3], F.c[4])
    @test couple_channels(F.cd, F.c, F.cd, F.c) == [(unit(FermionNumber), FermionNumber(1))]
    (; Sp, Sm, Sz) = spin_ops(Vu1, U1Irrep(1), U1Irrep(0))
    @test couple_channels(Sp, Sz, Sm) == couple_channels(Sp[1], Sz[2], Sm[3])
end

# A slot an operand does not act on is held by the pass-through letter and dropped on placement, so
# the placed bag is the one the placed spelling writes — scalar as a `K = 0` term, a one-sided
# identity factor as a shorter term.
@testset "pass-through slots" begin
    S = spin(Vsu2)
    Isu2 = Terms{SU2Irrep}

    @testset "SiteOperator conversion keeps the scalar part" begin
        for V in (Vsu2, Vu1, Vf)
            I = sectortype(V)
            O = project(randn(ComplexF64, V ← V), V) + 0.7
            B = LocalOperator(O)
            @test nsites(B) == 1
            @test B[4] == O[4]
            @test any(t -> arity(t) == 0, B[4])
            @test LocalOperator(scalarop(2.0, V))[3] == scalarop(2.0, V)[3]
            @test onlyterm(LocalOperator(scalarop(2.0, V))[3]) == onlyterm(one(Terms{I}))
            @test onlyterm(LocalOperator(scalarop(2.0, V))[3]).coeff == 2
        end
    end

    @testset "scalars on a LocalOperator" begin
        b = dot(S, S)
        placed = dot(S[1], S[2]) + one(Isu2) / 4
        @test (b + 1 / 4)[1] == placed
        @test (b + 1 / 4)[1] ≈ placed
        @test (1 / 4 + b)[1] ≈ placed
        @test (b - 1 / 4)[1] ≈ dot(S[1], S[2]) - one(Isu2) / 4
        @test (1 / 4 - b)[1] ≈ one(Isu2) / 4 - dot(S[1], S[2])
        @test (b + 1 / 4)[2, 5] ≈ dot(S[2], S[5]) + one(Isu2) / 4
        @test length(b + 1 / 4) == 2
        @test nsites(b + 1 / 4) == 2
        # the scalar occupies both slots unplaced, and none placed
        @test all(t -> arity(t) == 2, b + 1 / 4)
        @test sort!([arity(t) for t in (b + 1 / 4)[1]]) == [0, 2]

        # `one` is the all-pass-through term, and places as the K = 0 term
        @test nsites(one(b)) == 2
        @test length(one(b)) == 1
        @test one(b)[3] == one(Isu2)
        @test arity(onlyterm(one(b)[3])) == 0
        @test all(k -> ispassthrough(k.op), onlyterm(one(b).terms).keys)
        @test b + 1 / 4 == b + one(b) / 4
        @test iszero(b + 1 - one(b) - b)
    end

    @testset "a scalar part of an operand becomes a pass-through slot" begin
        (; Sp, Sm, Sz) = spin_ops(Vu1, U1Irrep(1), U1Irrep(0))
        Iu1 = Terms{U1Irrep}

        B = couple(Sz + 1 / 2, Sz)
        @test nsites(B) == 2
        @test length(B) == 6                      # 4 from Sz⊗Sz, 2 from 𝟙⊗Sz
        for (i, j) in ((1, 2), (2, 4))
            @test B[i, j] == couple(Sz[i], Sz[j]) + Sz[j] / 2
            @test B[i, j] ≈ couple(Sz[i], Sz[j]) + Sz[j] / 2
        end
        C = couple(Sz, Sz + 1 / 2)
        @test C[1, 3] ≈ couple(Sz[1], Sz[3]) + Sz[1] / 2
        D = couple(Sz + 1 / 2, Sz + 1 / 2)
        @test D[1, 2] ≈ couple(Sz[1], Sz[2]) + Sz[1] / 2 + Sz[2] / 2 + one(Iu1) / 4

        # a pass-through pair whose charges cannot fuse to `to` is dropped like any other
        @test couple(Sp + 1 / 2, Sm) == couple(Sp, Sm)
        @test couple(Sp + 1 / 2, Sm; to = U1Irrep(0))[1, 2] == couple(Sp[1], Sm[2])
        @test couple(Sp, Sm + 1 / 2; to = U1Irrep(1))[1, 2] == Sp[1] / 2
        # pass-through in the *first* slot: the charged letter lands on the second site alone
        @test couple(scalarop(1.0, Vu1), Sp; to = U1Irrep(1))[2, 5] == Sp[5]
        # pass-through everywhere: the scalar
        @test couple(scalarop(2.0, Vu1), scalarop(3.0, Vu1))[1, 2] == 6 * one(Iu1)
        @test onlyterm(couple(scalarop(2.0, Vu1), scalarop(3.0, Vu1))[1, 2]).coeff == 6

        # non-abelian: a pass-through slot next to a charged total
        @test couple(S + 1, S) == couple(S, S)                       # 𝟙 ⊗ S cannot reach 0
        E = couple(S + 1, S; to = one_)
        @test length(E) == 2
        @test E[1, 3] == couple(S[1], S[3]; to = one_) + S[3]
        @test E[1, 3] ≈ couple(S[1], S[3]; to = one_) + S[3]
    end

    # The case a unit-bond pass-through key would get wrong: inside a charged caterpillar the
    # running bond charge crossing the idle slot is the charge of what came before, not the unit
    # sector. Checked on the key itself and, placed, densely against the hand-placed operator.
    @testset "pass-through slot inside a charged caterpillar" begin
        (; Sp, Sm, Sz) = spin_ops(Vu1, U1Irrep(1), U1Irrep(0))
        α = 0.3
        B = couple(Sp, Sz + α, Sm)
        @test nsites(B) == 3
        @test length(B) == 3
        for t in B
            @test t.keys[1].bond == U1Irrep(1)
            @test t.keys[2].bond == U1Irrep(1)        # the running charge out of S⁺, not the unit
            @test total(t) == unit(U1Irrep)
        end
        @test count(t -> ispassthrough(t.keys[2].op), B) == 1
        @test only(filter(t -> ispassthrough(t.keys[2].op), collect(B))).coeff ≈ α

        ref(i, j, k) = couple(Sp[i], Sz[j], Sm[k]) + α * couple(Sp[i], Sm[k])
        for sites in ([1, 2, 3], [2, 3, 4], [1, 3, 4], [1, 2, 4])
            placed = B[sites...]
            @test placed == ref(sites...)
            @test placed ≈ ref(sites...)
            # the idle slot leaves no trace: the pass-through term is the two-site one
            @test sort!([arity(t) for t in placed]) == [2, 3, 3]
            lat = FiniteChain(Vu1, maximum(sites))
            @test densop(placed, lat) ≈ densop(ref(sites...), lat)
            @test islossless(opsum(lat, placed))
        end

        # fermions: the idle slot sits on an odd bond, so the string crosses it
        F = fermion_ops(Vf)
        β = 0.5
        Bf = couple(F.cd, F.n + β, F.c)
        @test nsites(Bf) == 3
        @test length(Bf) == 2
        for t in Bf
            @test t.keys[2].bond == FermionNumber(1)
        end
        reff(i, j, k) = couple(F.cd[i], F.n[j], F.c[k]) + β * couple(F.cd[i], F.c[k])
        for sites in ([1, 2, 3], [2, 3, 4], [1, 3, 4], [1, 2, 4])
            placed = Bf[sites...]
            @test placed == reff(sites...)
            @test placed ≈ reff(sites...)
            lat = FiniteChain(Vf, maximum(sites))
            @test densop(placed, lat) ≈ densop(reff(sites...), lat)
            # independent oracle: the dense block of the hand-placed operator, projected, then
            # reconstructed through the same dense map
            @test densop(placed, lat) ≈
                densop(project(instantiate(opsum(lat, reff(sites...))), collect(1:length(lat))), lat)
            @test islossless(opsum(lat, placed))
            mpo = irrep_mpo(opsum(lat, placed))
            @test mpo_tensormap(irrep_mpo_tensors(mpo, lat)) ≈ instantiate(opsum(lat, placed))
        end
        # and hermitian when completed with its partner, as a sanity check on the sign
        lat4 = FiniteChain(Vf, 4)
        Hh = opsum(Bf[1, 2, 4], opsum(lat4, Bf[1, 2, 4])'.terms)
        Oh = densop(Hh, lat4)
        @test Oh ≈ Oh'

        # SU(2): the pass-through slot carries the spin-1 running charge
        Bs = couple(S, scalarop(1.0, Vsu2) + S, S; to = zero_)
        @test length(Bs) == 2
        @test all(t -> t.keys[2].bond == one_, Bs)
        refs(i, j, k) = couple(S[i], S[j], S[k]) + couple(S[i], S[k])
        for sites in ([1, 2, 3], [1, 3, 4], [2, 3, 4])
            @test Bs[sites...] == refs(sites...)
            lat = FiniteChain(Vsu2, maximum(sites))
            @test densop(Bs[sites...], lat) ≈ densop(refs(sites...), lat)
        end
    end
end

@testset "chains from unplaced blocks are lossless" begin
    N = 5
    S = spin(Vsu2)
    (; Sp, Sm, Sz) = spin_ops(Vu1, U1Irrep(1), U1Irrep(0))
    F = fermion_ops(Vf)

    cases = (
        su2 = (
            Vsu2,
            dot(S, S) + 1 / 4, (i, j) -> dot(S[i], S[j]) + one(Terms{SU2Irrep}) / 4,
            couple(S, S, S), (i, j, k) -> couple(S[i], S[j], S[k]),
        ),
        u1 = (
            Vu1,
            couple(Sp, Sm) / 2 + couple(Sm, Sp) / 2 + 0.7 * couple(Sz + 1 / 2, Sz),
            (i, j) -> couple(Sp[i], Sm[j]) / 2 + couple(Sm[i], Sp[j]) / 2 +
                0.7 * (couple(Sz[i], Sz[j]) + Sz[j] / 2),
            couple(Sp, Sz + 0.3, Sm),
            (i, j, k) -> couple(Sp[i], Sz[j], Sm[k]) + 0.3 * couple(Sp[i], Sm[k]),
        ),
        fermion = (
            Vf,
            -(couple(F.cd, F.c) + couple(F.c, F.cd)) + 0.4 * couple(F.n - 1 / 2, F.n - 1 / 2),
            (i, j) -> -(couple(F.cd[i], F.c[j]) + couple(F.c[i], F.cd[j])) +
                0.4 * (couple(F.n[i], F.n[j]) - F.n[i] / 2 - F.n[j] / 2 + one(Terms{FermionNumber}) / 4),
            couple(F.cd, F.n + 0.5, F.c),
            (i, j, k) -> couple(F.cd[i], F.n[j], F.c[k]) + 0.5 * couple(F.cd[i], F.c[k]),
        ),
    )

    for (label, (V, B2, ref2, B3, ref3)) in pairs(cases)
        @testset "$label" begin
            lat = FiniteChain(V, N)
            H = opsum(
                (B2[i] for i in 1:(N - 1)),
                (0.3 * B2[i, i + 2] for i in 1:(N - 2)),
                (0.1 * B3[i] for i in 1:(N - 2)),
                0.2 * B3[1, 3, 5],
            )
            Href = opsum(
                (ref2(i, i + 1) for i in 1:(N - 1)),
                (0.3 * ref2(i, i + 2) for i in 1:(N - 2)),
                (0.1 * ref3(i, i + 1, i + 2) for i in 1:(N - 2)),
                0.2 * ref3(1, 3, 5),
            )
            @test H == Href
            @test H ≈ Href
            @test islossless(opsum(lat, H))
            mpo = irrep_mpo(opsum(lat, H))
            @test mpo_tensormap(irrep_mpo_tensors(mpo, lat)) ≈ instantiate(opsum(lat, H))
            @test densop(H, lat) ≈ densop(Href, lat)
        end
    end
end

@testset "errors" begin
    S = spin(Vsu2)
    (; Sp, Sm, Sz) = spin_ops(Vu1, U1Irrep(1), U1Irrep(0))
    F = fermion_ops(Vf)

    @testset "mixing placed and unplaced operands" begin
        for f in (
                () -> couple(S, S[2]), () -> couple(S[1], S), () -> couple(S, S, S[3]),
                () -> couple(S[1], S[2], S), () -> couple(dot(S, S), S[3]),
                () -> couple(S, onlyterm(S[2])), () -> dot(S, S[2]), () -> dot(S[1], S),
                () -> dot(dot(S, S), S[3]), () -> couple_channels(S, S[2]),
                () -> couple_channels(S[1], S, S),
            )
            @test_throws ArgumentError f()
            msg = errmsg(f)
            @test occursin("unplaced", msg) && occursin("placed", msg)
        end
        # an all-`Term` call is still a `MethodError`, as before
        @test_throws MethodError couple(onlyterm(S[1]), onlyterm(S[2]))
        @test_throws MethodError dot(onlyterm(S[1]), onlyterm(S[2]))
    end

    @testset "forced-channel errors are the placed ones" begin
        @test_throws ArgumentError couple(S, S, S, S)
        @test errmsg(() -> couple(S, S, S, S)) == errmsg(() -> couple(S[1], S[2], S[3], S[4]))
        @test occursin("genuine choice", errmsg(() -> couple(S, S, S, S)))
        @test_throws ArgumentError couple(S, S, S; to = SU2Irrep(5))
        @test errmsg(() -> couple(S, S, S; to = SU2Irrep(5))) ==
            errmsg(() -> couple(S[1], S[2], S[3]; to = SU2Irrep(5)))
        @test_throws ArgumentError couple(F.cd, F.cd, F.cd, F.c)
        @test errmsg(() -> couple(F.cd, F.cd, F.cd, F.c)) ==
            errmsg(() -> couple(F.cd[1], F.cd[2], F.cd[3], F.c[4]))
        # two operands that cannot reach `to`
        @test_throws ArgumentError couple(Sp, Sp)
        @test errmsg(() -> couple(Sp, Sp)) == errmsg(() -> couple(Sp[1], Sp[2]))
        @test_throws ArgumentError couple(S, S; via = :tree)
        @test_throws ArgumentError couple(S, S, S; via = :tree)
        @test_throws ArgumentError couple_channels(S)
    end

    @testset "a multi-slot operand after the first" begin
        b = couple(S, S; to = one_)
        @test_throws ArgumentError couple(S, b)
        @test_throws ArgumentError couple(S, S, b)
        @test occursin("slots", errmsg(() -> couple(S, b)))
        @test_throws ArgumentError couple_channels(S, b)
        # as the first operand it is fine
        @test nsites(couple(b, S)) == 3
        @test couple(b, S)[1] == couple(couple(S[1], S[2]; to = one_), S[3])
    end

    @testset "dot keeps its single-letter rule" begin
        @test_throws ArgumentError dot(S + 1, S)
        @test_throws ArgumentError dot(Sz, Sz)                  # two letters
        @test_throws ArgumentError dot(dot(S, S), S)            # two slots
        @test_throws ArgumentError dot(Sp, Sp)                  # charges do not fuse to unit
        @test_throws ArgumentError dot(scalarop(1.0, Vsu2), S)
    end
end
