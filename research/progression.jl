# The operator progression as executable code, no prose.
#
# One block per step of research/examples-progression.md §3. The point is to read the *calls* in
# order: what you import, how an on-site operator is named, how it is placed, where the lattice
# appears. Run it to confirm every step works; read it to judge the interface.
#
#   julia --project research/progression.jl

using OpSum
using OpSum: IrrepTensorOperators
using TensorKit
using MatrixAlgebraKit: truncrank
using LinearAlgebra: dot, norm, eigvals

report(label, H) = let m = mpo(H)
    D = maximum(length, m.bondsectors)
    Dd = maximum(s -> sum(dim, s), m.bondsectors)
    ok = islossless(H)
    println(
        rpad(label, 42), " nterms=", lpad(length(H), 5),
        "  D=", lpad(D, 3), "  D_dense=", lpad(Dd, 4), "  lossless=", ok
    )
    return m
end

# ── Tier 0 · the whole pipeline once ────────────────────────────────────────────────────────────
B½ = spin_basis(SU2Irrep)                        # a site is a basis of operators ...
S½ = spin_ops(SU2Irrep)                          # ... and the builder expands the common ones in it

lat = FiniteChain(B½, 2)                         # the lattice ...
H0 = opsum(lat, S½.S_exchange[1, 2])             # ... enters with the terms: an OperatorSum
W0 = report("0.1 one bond", H0)
Ws, secs = W0.Ws, W0.bondsectors                 # a SymbolicMPO's reduced data
@assert mpo_terms(Ws, secs) ≈ H0.terms           # faithfulness compares latticeless term bags
@assert mpo_tensormap(irrep_mpo_tensors(W0)) ≈ instantiate(H0)

# ── Tier 1 · the on-site operators, abelian ────────────────────────────────────────────────────
# 1.1 trivial sector: write the matrices down and project them into the basis
Bc = spin_basis()
Vc = space(Bc)
σx = project(TensorMap(ComplexF64[0 1; 1 0], Vc ← Vc), Bc)
σz = project(TensorMap(ComplexF64[1 0; 0 -1], Vc ← Vc), Bc)
tfim(N; J = 1.0, g = 0.5) = opsum(
    FiniteChain(Bc, N),
    (-J * σz[i] * σz[i + 1] for i in 1:(N - 1)),
    (-g * σx[i] for i in 1:N),                   # K=1 and K=2 mixed in one operator
)
report("1.1 transverse-field Ising (trivial)", tfim(6))

# 1.2 U(1): the builder has the neutral operators; Sᶻ is one letter, and `*` is the product
Bu = spin_basis(U1Irrep)
Su = spin_ops(U1Irrep)
@assert length(Su.S_z) == 1
@assert opsum(FiniteChain(Bu, 3), Su.S_z[i] * Su.S_z[i + 1] for i in 1:2) ≈
    opsum(FiniteChain(Bu, 3), Su.S_z_S_z[i, i + 1] for i in 1:2)
xxz(N; Δ = 1.0) = opsum(
    FiniteChain(Bu, N),
    Su.S_plus_S_min[i, i + 1] / 2 + Su.S_min_S_plus[i, i + 1] / 2 +
        Δ * Su.S_z[i] * Su.S_z[i + 1] for i in 1:(N - 1)
)
report("1.2 XXZ (U(1))", xxz(6))

# ── Tier 2 · non-abelian ───────────────────────────────────────────────────────────────────────
heisenberg(N) = opsum(FiniteChain(B½, N), S½.S_exchange[i, i + 1] for i in 1:(N - 1))
report("2.1 Heisenberg (SU(2))", heisenberg(6))
# same operator, two symmetries: compare spectra, not MPOs
blockspec(H) = let O = instantiate(H)
    Oop = numind(O) == 2 * numout(O) ? O : removeunit(O, numind(O))
    sort!(reduce(vcat, [repeat(real(eigvals(Matrix(b))), dim(c)) for (c, b) in blocks(Oop)]))
end
@assert blockspec(heisenberg(6)) ≈ blockspec(xxz(6))

# 2.2 biquadratic: (S·S)² is the product of two placed two-site operators on the same sites
B1 = spin_basis(SU2Irrep; spin = 1)
S1 = spin_ops(SU2Irrep; spin = 1)
akltish(N; β = 1 / 3) = opsum(
    FiniteChain(B1, N),
    S1.S_exchange[i, i + 1] + β * S1.S_exchange[i, i + 1] * S1.S_exchange[i, i + 1]
        for i in 1:(N - 1)
)
report("2.2 bilinear-biquadratic (spin-1)", akltish(4))

# ── Tier 3 · arity: products of operators ──────────────────────────────────────────────────────
# 3.1 three sites: two bonds sharing a site (the shared site is an on-site product, resolved in the
# basis when the term enters the OperatorSum)
ex = S½.S_exchange
report("3.1 three-body (S₁·S₂)(S₂·S₃)", opsum(FiniteChain(B½, 3), ex[1, 2] * ex[2, 3]))
# 3.2 four sites: the three pairings are different operators, placement may be in any site order
pairings = (ex[1, 2] * ex[3, 4], ex[1, 3] * ex[2, 4], ex[1, 4] * ex[2, 3])
for (k, p) in enumerate(pairings)
    report("3.2 four-body pairing $k", opsum(FiniteChain(B½, 4), p))
end
# 3.3 fermions: the sign of an interleaved product is the braiding, so the factors commute
Bf = fermion_basis(U1Irrep)
F = fermion_ops(U1Irrep)
latf4 = FiniteChain(Bf, 4)
@assert opsum(latf4, F.f_plus_f_min[1, 3] * F.f_plus_f_min[2, 4]) ≈
    opsum(latf4, F.f_plus_f_min[2, 4] * F.f_plus_f_min[1, 3])
report("3.3 four-fermion c†₁c₂ c†₃c₄", opsum(latf4, F.f_plus_f_min[1, 2] * F.f_plus_f_min[3, 4]))

# ── Tier 4 · fermions ──────────────────────────────────────────────────────────────────────────
# 4.1 hopping: placement supplies the anticommutation sign; `H'` needs the bases, so it lives on
# the OperatorSum
latf = FiniteChain(Bf, 6)
fwd = opsum(latf, -1.0 * F.f_plus_f_min[i, i + 1] for i in 1:5)
hop = fwd + fwd'
@assert hop ≈ opsum(
    latf,
    -1.0 * (F.f_plus_f_min[i, i + 1] + F.f_plus_f_min[i + 1, i]) for i in 1:5
)
report("4.1 free fermions", hop)
# `copy` + `opsum!` is the idiom for a variant of an existing operator; `+` would copy anyway, but
# folding it would be quadratic.
tV = opsum!(copy(hop), (2.0 * F.f_num[i] * F.f_num[i + 1] for i in 1:5))
report("4.2 t-V chain", tV)

# 4.3 Kitaev: pairing breaks U(1) but keeps parity, so grade by FermionParity alone
Bp = fermion_basis()
Fp = fermion_ops()
kitaev(N; t = 1.0, Δ = 0.5) = opsum(
    FiniteChain(Bp, N),
    (-t * (Fp.f_plus_f_min[i, i + 1] + Fp.f_plus_f_min[i + 1, i]) for i in 1:(N - 1)),
    (Δ * (Fp.f_plus_f_plus[i, i + 1] + Fp.f_min_f_min[i + 1, i]) for i in 1:(N - 1)),
)
report("4.3 Kitaev chain (parity only)", kitaev(6))

# 4.4 non-abelian *and* fermionic: the Hubbard site, U(1) charge x SU(2) spin x parity. `e_hopping`
# (the spin-summed hop and its conjugate) and `ud_num` (n↑n↓) are single elements of the basis, so
# no Clebsch–Gordan factor is written by hand. A generic dense block on the same site goes through
# `project(h, B)`, to exercise the projection on a block that is not a Hamiltonian.
Bh = hubbard_basis(U1Irrep, SU2Irrep)
Hh = hubbard_ops(U1Irrep, SU2Irrep)
hubbard_su2(N; t = 1.0, U = 4.0) = opsum(
    FiniteChain(Bh, N),
    (-t * Hh.e_hopping[i, i + 1] for i in 1:(N - 1)),
    (U * Hh.ud_num[i] for i in 1:N),
)
report("4.4 SU(2) Hubbard chain", hubbard_su2(4))
Vh = space(Bh)
hubblock = project(randn(ComplexF64, Vh ⊗ Vh ← Vh ⊗ Vh), Bh)
report("4.5 SU(2) Hubbard site, random block", opsum(FiniteChain(Bh, 4), hubblock[i] for i in 1:3))

# ── Tier 5 · the lattice itself ────────────────────────────────────────────────────────────────
# a mixed bond has no builder: it is diagonal in the total spin, so give it per coupled sector
function exchange(Ba, Bb)
    Va, Vb = space(Ba), space(Bb)
    sa, sb = only(sectors(Va)).j, only(sectors(Vb)).j
    t = zeros(ComplexF64, Va ⊗ Vb ← Va ⊗ Vb)
    for (J, blk) in blocks(t)
        blk .= (J.j * (J.j + 1) - sa * (sa + 1) - sb * (sb + 1)) / 2
    end
    return project(t, [Ba, Bb])
end
alt = FiniteChain([B½, B1, B½, B1])                # the case FiniteChain(B, N) cannot express
report(
    "5.1 alternating spin-½ / spin-1",
    opsum(alt, (isodd(i) ? exchange(B½, B1) : exchange(B1, B½))[i, i + 1] for i in 1:3),
)

# ── Tier 6 · geometry ──────────────────────────────────────────────────────────────────────────
siteindex(x, y, Ly) = (x - 1) * Ly + mod1(y, Ly)
function cyl_bonds(Lx, Ly)
    bs = Tuple{Int, Int}[]
    for x in 1:Lx, y in 1:Ly
        Ly > 2 && push!(bs, minmax(siteindex(x, y, Ly), siteindex(x, y + 1, Ly)))
        x < Lx && push!(bs, minmax(siteindex(x, y, Ly), siteindex(x + 1, y, Ly)))
    end
    return unique!(bs)
end
report(
    "6.1 Heisenberg cylinder 4x3",
    opsum(FiniteChain(B½, 12), ex[i, j] for (i, j) in cyl_bonds(4, 3)),
)

# ── Tier 7 · long range and truncation ─────────────────────────────────────────────────────────
powerlaw(N; α = 2.0) = opsum(
    FiniteChain(B½, N),
    abs(m - n)^(-α) * ex[n, m] for n in 1:(N - 1) for m in (n + 1):N
)
report("7.1 power law α=2, N=10", powerlaw(10))
let H6 = powerlaw(6), oracle = instantiate(H6), l6 = H6.lattice
    for k in (8, 4, 2)
        m = mpo(H6, SVDBondAlgorithm(truncrank(k)))
        Wt, st = m.Ws, m.bondsectors
        err = norm(mpo_tensormap(irrep_mpo_tensors(Wt, st, l6)) - oracle) / norm(oracle)
        println(
            "     truncrank($k): D_dense=", maximum(s -> sum(dim, s), st),
            "  rel.err=", round(err; sigdigits = 3)
        )
    end
end

# ── Tier 8 · infinite chains ───────────────────────────────────────────────────────────────────
# the *same* container and call; only the lattice and the return type change
inf1 = mpo(opsum(InfiniteChain(B½), ex[1, 2]))
println(rpad("8.1 infinite Heisenberg (L=1)", 42), " ", inf1)
inf2 = mpo(
    opsum(InfiniteChain([B½, B½]), 0.6 * ex[1, 2] + 1.4 * ex[2, 3]),
)
println(rpad("8.2 dimerised, L=2", 42), " ", inf2)
# 8.3 exponential decay: finite and infinite, same operator
hexp = (ex[1, 2], expterm(ex[1, 2]; decay = 0.4))
println(rpad("8.3 exp decay, FiniteChain", 42), " ", mpo(opsum(FiniteChain(B½, 8), hexp)))
println(rpad("8.3 exp decay, InfiniteChain", 42), " ", mpo(opsum(InfiniteChain(B½), hexp)))

# ── Tier 9 · operators that are not Hamiltonians ───────────────────────────────────────────────
lat5 = FiniteChain(Bu, 5)
report("9.1 total magnetisation Σ Sᶻ", opsum(lat5, Su.S_z[i] for i in 1:5))
report("9.2 correlator Sᶻ₁Sᶻ₄", opsum(lat5, Su.S_z[1] * Su.S_z[4]))
# 9.3 a *charged* operator: the builders only hand out neutral ones, so write the tensor with a
# charge leg and project it. Fine for mpo, refused by the Jordan emission
Vf = space(Bf)
occ = first(filter(!isone, collect(sectors(Vf))))         # the occupied sector
cdag = zeros(ComplexF64, Vf ← Vf ⊗ Vect[sectortype(Vf)](occ => 1))
for (_, blk) in blocks(cdag)
    blk .= 1
end
chg = opsum(FiniteChain(Bf, 4), project(cdag, Bf)[2])
report("9.3 charged: c†₂", chg)
println(
    "     jordan_mpo_tensors on it: ",
    try
        (jordan_mpo_tensors(chg); "accepted")
    catch
        "refused"
    end
)
# 9.4 a string: every site of a run is active
report("9.4 string n₁n₂n₃n₄", opsum(FiniteChain(Bf, 4), F.f_num[1] * F.f_num[2] * F.f_num[3] * F.f_num[4]))

# ── Tier 10 · handing it onward ────────────────────────────────────────────────────────────────
Wj = jordan_mpo_tensors(heisenberg(6))
println(rpad("10.2 Jordan form bond sizes", 42), " ", map(W -> size(W, 4), Wj))
Ti = irrep_mpo_tensors(inf1)
println(
    rpad("10.3 infinite tensors tile", 42), " ",
    space(Ti[1], 1) == space(Ti[end], 4)'
)

println("\nPROGRESSION OK")
