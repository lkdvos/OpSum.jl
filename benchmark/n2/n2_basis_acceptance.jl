# Acceptance check for #38: the N₂ Hamiltonians, compressed in
# matrix units and in an explicit basis of the SU(2) Hubbard site.
#
# The matrix-unit side is the bare sweep over the same terms. Energies are checked against an exact
# spectrum in `n2_spinfree_acceptance.jl`.
#
# Run from OpSum's test environment (it has MPSKit), with the #38 material checked out next to it:
#
#     jld --test run benchmark/n2/n2_basis_acceptance.jl            # STO-3G
#     N2_BASIS=CCPVDZ jld --test run benchmark/n2/n2_basis_acceptance.jl
#
# The terms come from `opsum_qchem_su2.jl` (MPSKitExperimental.jl, PR maartenvd/MPSKitExperimental.jl#1),
# which writes them in matrix-unit letters on its own site space `U₁ ⊠ SU₂ ⊠ fℤ₂`. The basis is built on
# that same space from neutral pairs written with the same machinery, so both share its conventions,
# and the terms are re-expressed in it letter by letter (`OpSum._rebasis`), which is exact because the
# basis spans every operator on the site.

using OpSum, TensorKit, LinearAlgebra, Printf
using OpSum: Term, Terms, canonicalize!, _rebasis
using TensorKit: removeunit, numind

const EXP = get(ENV, "MPSKITEXPERIMENTAL", "/mnt/home/ldevos/Projects/MPSKitExperimental.jl")
include(joinpath(EXP, "examples", "opsum_qchem_su2.jl"))
const Q = OpSumQChemSU2
const Vh = Q.Vh

# FCIDUMP: a line `v i j k l` with `i, k > 0` is `(ij|kl)` (8-fold symmetric), with `k = l = 0` it is
# `h_ij`, and all zeros is `E₀`.
function read_fcidump(path)
    text = read(path, String)
    norb = parse(Int, match(r"NORB\s*=\s*(\d+)", text)[1])
    body = split(text, r"&END|/\s*\n"; limit = 2)[2]
    h = zeros(norb, norb)
    eri = zeros(norb, norb, norb, norb)
    E0 = 0.0
    for line in eachsplit(body, '\n')
        f = split(line)
        length(f) == 5 || continue
        v = parse(Float64, replace(f[1], r"[dD]" => "e"))
        i, j, k, l = parse.(Int, f[2:5])
        if i == j == k == l == 0
            E0 = v
        elseif k == 0 && l == 0
            h[i, j] = h[j, i] = v
        else
            for (a, b, c, d) in (
                    (i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                    (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i),
                )
                eri[a, b, c, d] = v
            end
        end
    end
    return E0, h, eri
end

# A term bag from `(coefficient, string)`s, strings as in `opsum_qchem_su2.jl`: `(site, :cd | :c, spin
# label)` left to right, summed over the spin labels.
function strings_terms(strings)
    acc = Dict{Tuple{Vector{Int}, Vector{Int}, Vector{Q.S2}}, Float64}()
    for (c, s) in strings
        Q.add_string!(acc, c, s)
    end
    return opsum(Q.build_term(k..., c) for (k, c) in acc if abs(c) > 1.0e-14)
end
densen(ts, n) = (t = OpSum.dense(ts, fill(Vh, n)); removeunit(t, numind(t)))

# The Hubbard site spanned by neutral operators: `n`, `nꜛꜜ` on one site, and pairs whose pieces are
# `a†`, `a` (hopping), `S⃗` (exchange), `Δ†`, `Δ` (pair hopping) and `n·a†`, `n·a` (correlated hopping).
function qchem_basis()
    one(s) = densen(strings_terms([(1.0, s)]), 1)
    two(s) = densen(strings_terms([(1.0, s)]), 2)
    return Basis(
        [
            one([(1, :cd, 1), (1, :c, 1)]),                                  # n
            one([(1, :cd, 1), (1, :cd, 2), (1, :c, 2), (1, :c, 1)]),         # 2 nꜛnꜜ
            two([(1, :cd, 1), (2, :c, 1)]),                                  # a†₁ a₂
            two([(2, :cd, 1), (1, :c, 1)]),                                  # a₁ a†₂
            two([(1, :cd, 1), (1, :c, 2), (2, :cd, 2), (2, :c, 1)]),         # exchange
            two([(1, :cd, 1), (1, :cd, 2), (2, :c, 2), (2, :c, 1)]),         # Δ†₁ Δ₂
            two([(1, :cd, 1), (1, :cd, 2), (1, :c, 2), (2, :c, 1)]),         # (a†n)₁ a₂
            two([(2, :cd, 1), (1, :cd, 2), (1, :c, 2), (1, :c, 1)]),         # a†₂ (n a)₁
        ]
    )
end

centre(secs, b) = (length(secs[b]), sum(dim, secs[b]))

function run(which)
    E0, h, eri = read_fcidump(joinpath(EXP, "examples", "N2.$which.FCIDUMP"))
    N = size(h, 1)
    t0 = time()
    terms, ntuples = Q.qchem_opsum_su2(h, eri)
    @printf "%s: %d orbitals, %d term tuples, built in %.1f s\n" which N ntuples (time() - t0)

    B = qchem_basis()
    @printf "basis: %s\n" B
    lat = FiniteChain(B, N)
    rb = canonicalize!(_rebasis(canonicalize!(terms).terms, (t, j) -> lat[t.sites[j]]; placed = true)).terms
    # different matrix-unit tuples land on the same basis term and cancel to round-off there
    cmax = maximum(t -> abs(t.coeff), rb)
    Hb = opsum(lat, Terms(filter(t -> abs(t.coeff) > 1.0e-12 * cmax, rb)))
    tt = OpSum.TermTable(terms, N)
    @printf "terms: %d in matrix units, %d in the basis\n" OpSum.nterms(tt) length(Hb.terms)

    secs_m = OpSum._sweep_channels(tt, N, OpSum.VertexCover())[2]
    secs_b = mpo(Hb).bondsectors
    bonds = which == "STO3G" ? [N ÷ 2] : [7, 14, 21]
    for b in bonds
        @printf "bond %2d: matrix units %4d / %4d   basis %4d / %4d   (channels / dims)\n" b centre(secs_m, b)... centre(secs_b, b)...
    end

    return nothing
end

get(ENV, "N2_NORUN", "") == "1" || run(get(ENV, "N2_BASIS", "STO3G"))
