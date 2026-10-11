# Acceptance check for step 5 (operator-bases.md §3.6, handoff §5): the #38 N₂ Hamiltonians written
# spin-free, from the spin-summed excitation operator `E[p, q] = Σσ a†_pσ a_qσ` alone,
#
#     H = Σ h[p, q] E[p, q] + ½ Σ (pq|rs) (E[p, q] * E[r, s] - δ(q, r) E[p, s]),
#
# with no Clebsch–Gordan code: every `*` multiplies neutral operators, out-of-order legs are re-coupled
# into site order and coinciding indices are on-site products resolved in the basis.
#
# Checked two ways:
#
# * against an exact spectrum, on sub-blocks of 4–5 orbitals: a plain Jordan–Wigner Fock-space matrix
#   built here from `h` and `eri` (no TensorKit, no OpSum);
# * against the matrix-unit Hamiltonian of `opsum_qchem_su2.jl` re-expressed in the same basis
#   (`n2_basis_acceptance.jl`), at full size: the same term set and the same bond dimensions.
#   Not the coefficients: that script's two-site strings carry the opposite sign of the physical
#   `a†_p a_q` under the current OpSum conventions (its per-letter sign fix predates the graded sign of
#   #36), so its Hamiltonian misses the exact spectrum on a generic sub-block (orbitals 1:5 of STO-3G
#   by 1.5e-2). `E` is therefore minus its two-site string, which is what makes `E[p, p] = n_p`.
#
#     jld --test run research/n2_spinfree_acceptance.jl             # STO-3G
#     N2_BASIS=CCPVDZ jld --test run research/n2_spinfree_acceptance.jl

ENV["N2_NORUN"] = "1"
include(joinpath(@__DIR__, "n2_basis_acceptance.jl"))

function spinfree(lat, E, h, eri)
    N = size(h, 1)
    out = Terms{sectortype(lat[1])}()
    for p in 1:N, q in 1:N
        abs(h[p, q]) > 1.0e-14 && append!(out, h[p, q] * E[p, q])
    end
    for p in 1:N, q in 1:N, r in 1:N, s in 1:N
        v = eri[p, q, r, s]
        abs(v) > 1.0e-14 || continue
        append!(out, (v / 2) * (E[p, q] * E[r, s]))
        q == r && append!(out, (-v / 2) * E[p, s])
    end
    return opsum(lat, out)
end

# the exact spectrum, in a Jordan–Wigner Fock space of spin orbitals (p, σ)
function fock_spectrum(h, eri)
    N = size(h, 1)
    M = 2N
    a1, Z, Id2 = [0.0 1.0; 0.0 0.0], [1.0 0.0; 0.0 -1.0], Matrix(1.0I, 2, 2)
    ann(k) = foldl(kron, [j < k ? Z : j == k ? a1 : Id2 for j in 1:M])
    A = [ann(2(p - 1) + σ) for p in 1:N, σ in 1:2]
    H = zeros(2^M, 2^M)
    for p in 1:N, q in 1:N, σ in 1:2
        H += h[p, q] * A[p, σ]' * A[q, σ]
    end
    for p in 1:N, q in 1:N, r in 1:N, s in 1:N
        v = eri[p, q, r, s]
        iszero(v) && continue
        for σ in 1:2, τ in 1:2
            H += (v / 2) * A[p, σ]' * A[r, τ]' * A[s, τ] * A[q, σ]
        end
    end
    return sort(eigvals(Hermitian(H)))
end

function tk_spectrum(H)
    t = instantiate(H)
    t = removeunit(t, numind(t))
    ev = Float64[]
    for (c, b) in blocks(t)
        append!(ev, repeat(real(eigvals(b)), dim(c)))
    end
    return sort(ev)
end

function basis_terms(lat, h, eri)
    terms, _ = Q.qchem_opsum_su2(h, eri)
    rb = canonicalize!(_rebasis(canonicalize!(terms).terms, (t, j) -> lat[t.sites[j]]; placed = true)).terms
    cmax = maximum(t -> abs(t.coeff), rb)
    return opsum(lat, Terms(filter(t -> abs(t.coeff) > 1.0e-12 * cmax, rb)))
end

function run_spinfree(which)
    E0, h, eri = read_fcidump(joinpath(EXP, "examples", "N2.$which.FCIDUMP"))
    N = size(h, 1)
    B = hubbard_basis()
    E = -project(densen(strings_terms([(1.0, [(1, :cd, 1), (2, :c, 1)])]), 2), B)

    for orbs in (1:4, 1:5, [1, 4, 7, 9, 10])
        o = collect(orbs)
        Hs = spinfree(FiniteChain(B, length(o)), E, h[o, o], eri[o, o, o, o])
        @printf "orbitals %-16s spin-free vs exact spectrum: %.2e\n" string(orbs) maximum(abs, tk_spectrum(Hs) - fock_spectrum(h[o, o], eri[o, o, o, o]))
    end

    lat = FiniteChain(B, N)
    t0 = time()
    Hs = spinfree(lat, E, h, eri)
    @printf "%s: spin-free H on %d orbitals built in %.1f s, %d terms\n" which N (time() - t0) length(Hs.terms)
    Hb = basis_terms(lat, h, eri)
    @printf "same term set as opsum_qchem_su2.jl in the basis (%d terms): %s\n" length(Hb.terms) (Set(Hs.terms) == Set(Hb.terms))

    secs_s, secs_b = mpo(Hs).bondsectors, mpo(Hb).bondsectors
    for b in (which == "STO3G" ? [N ÷ 2] : [7, 14, 21])
        @printf "bond %2d: opsum_qchem_su2.jl %4d / %4d   spin-free %4d / %4d   (channels / dims)\n" b centre(secs_b, b)... centre(secs_s, b)...
    end
    return nothing
end

run_spinfree(get(ENV, "N2_BASIS", "STO3G"))
