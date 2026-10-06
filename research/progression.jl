# The operator progression as executable code, no prose.
#
# One block per step of research/examples-progression.md §3. The point is to read the *calls* in
# order: what you import, how an on-site operator is named, how it is placed, where the lattice
# appears. Run it to confirm every step works; read it to judge the interface.
#
#   julia --project research/progression.jl

using OpSum
using OpSum: IrrepTensorOperators, instantiate, irrep_mpo, irrep_mpo_tensors, mpo_terms,
    mpo_tensormap, islossless, jordan_mpo_tensors, opsum, couple, couple_channels, spin, spin_ops, fermion_ops,
    matrixunit, scalarop, project, expterm, OperatorSum, opsum!, FiniteChain, InfiniteChain
using TensorKit
using TensorKit: FermionParity, FermionNumber, U1Irrep, SU2Irrep, removeunit, numind
using MatrixAlgebraKit: truncrank
using LinearAlgebra: dot, norm

report(label, H) = let m = irrep_mpo(H)
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
V½ = SU2Space(1 // 2 => 1)
S = spin(V½)

lat = FiniteChain(V½, 2)                         # the lattice ...
H0 = opsum(lat, dot(S[1], S[2]))                 # ... enters with the terms: an OperatorSum
mpo = report("0.1 one bond", H0)
Ws, secs = mpo                                   # FiniteMPO destructures as the pair
@assert mpo_terms(Ws, secs) ≈ H0.terms           # faithfulness compares latticeless term bags
@assert mpo_tensormap(irrep_mpo_tensors(mpo, lat)) ≈ instantiate(H0)

# ── Tier 1 · the on-site alphabet, abelian ─────────────────────────────────────────────────────
# 1.1 trivial sector: write the matrices down and project them
Vc = ℂ^2
σx = project(TensorMap(ComplexF64[0 1; 1 0], Vc ← Vc), Vc)
σz = project(TensorMap(ComplexF64[1 0; 0 -1], Vc ← Vc), Vc)
tfim(N; J = 1.0, g = 0.5) = opsum(
    FiniteChain(Vc, N),
    (-J * couple(σz[i], σz[i + 1]) for i in 1:(N - 1)),
    (-g * σx[i] for i in 1:N),                   # K=1 and K=2 mixed in one operator
)
report("1.1 transverse-field Ising (trivial)", tfim(6))

# 1.2 U(1): matrix units, then the builder; Sᶻ is composite (two letters)
Vu = Rep[U₁](0 => 1, 1 => 1)
up, dn = U1Irrep(1), U1Irrep(0)
Sp, Sm, Sz = matrixunit(Vu, up, dn), matrixunit(Vu, dn, up),
    (matrixunit(Vu, up, up) - matrixunit(Vu, dn, dn)) / 2
@assert spin_ops(Vu, up, dn).Sz ≈ Sz             # the builder agrees
@assert length(Sz) == 2
xxz(N; Δ = 1.0) = opsum(
    FiniteChain(Vu, N),
    couple(Sp[i], Sm[i + 1]) / 2 + couple(Sm[i], Sp[i + 1]) / 2 +
        Δ * couple(Sz[i], Sz[i + 1]) for i in 1:(N - 1)
)
report("1.2 XXZ (U(1))", xxz(6))

# ── Tier 2 · non-abelian ───────────────────────────────────────────────────────────────────────
heisenberg(N) = opsum(FiniteChain(V½, N), dot(S[i], S[i + 1]) for i in 1:(N - 1))
report("2.1 Heisenberg (SU(2))", heisenberg(6))
# same operator, two symmetries: compare spectra, not MPOs
blockspec(H) = let O = instantiate(H)
    Oop = numind(O) == 2 * numout(O) ? O : removeunit(O, numind(O))
    sort!(reduce(vcat, [repeat(real(eigvals(Matrix(b))), dim(c)) for (c, b) in blocks(Oop)]))
end
using LinearAlgebra: eigvals
@assert blockspec(heisenberg(6)) ≈ blockspec(xxz(6))

# 2.2 biquadratic: (S·S)² is an on-site *product*, so build the block and project it
V1 = SU2Space(1 => 1)
S1 = spin(V1)
bond = removeunit(instantiate(opsum([V1, V1], dot(S1[1], S1[2]))), 5)   # V⊗V ← V⊗V
akltish(N; β = 1 / 3) = let B = project(bond + β * (bond * bond))   # a LocalOperator: unplaced
    opsum(FiniteChain(V1, N), B[i] for i in 1:(N - 1))              # placed once per bond
end
report("2.2 bilinear-biquadratic (spin-1)", akltish(4))

# ── Tier 3 · arity and fusion channels ─────────────────────────────────────────────────────────
# 3.1 non-abelian, but the singlet forces j₁₂ = 1 — so the variadic form folds it too
@assert couple_channels(S[1], S[2], S[3]; to = SU2Irrep(0)) == [(SU2Irrep(1),)]
chirality = couple(S[1], S[2], S[3])
@assert chirality ≈ couple(couple(S[1], S[2]; to = SU2Irrep(1)), S[3]; to = SU2Irrep(0))
report("3.1 three-body chirality", opsum(FiniteChain(V½, 3), chirality))
# 3.2 four-body: two inner lines, and here the channels are a genuine choice (three of them), so
# the variadic form refuses and `couple_channels` is how you enumerate them
@assert length(couple_channels(S[1], S[2], S[3], S[4])) == 3
plaquette = couple(
    couple(couple(S[1], S[2]; to = SU2Irrep(0)), S[3]; to = SU2Irrep(1)),
    S[4]; to = SU2Irrep(0),
)
report("3.2 four-body plaquette", opsum(FiniteChain(V½, 4), plaquette))
# 3.3 abelian: every channel is forced by construction, so the whole chain folds variadically
Vf = Vect[FermionNumber](0 => 1, 1 => 1)
F = fermion_ops(Vf)
report(
    "3.3 four-fermion (variadic)",
    opsum(FiniteChain(Vf, 4), couple(F.cd[1], F.c[2], F.cd[3], F.c[4])),
)

# ── Tier 4 · fermions ──────────────────────────────────────────────────────────────────────────
# 4.1 hopping: `couple` supplies the anticommutation sign; `H'` needs the spaces, so it lives on
# the OperatorSum
latf = FiniteChain(Vf, 6)
fwd = opsum(latf, -1.0 * couple(F.cd[i], F.c[i + 1]) for i in 1:5)
hop = fwd + fwd'
@assert hop ≈ opsum(
    latf,
    -1.0 * (couple(F.cd[i], F.c[i + 1]) + couple(F.cd[i + 1], F.c[i])) for i in 1:5
)
report("4.1 free fermions", hop)
# `copy` + `opsum!` is the idiom for a variant of an existing operator; `+` would copy anyway, but
# folding it would be quadratic.
tV = opsum!(copy(hop), (2.0 * couple(F.n[i], F.n[i + 1]) for i in 1:5))
report("4.2 t-V chain", tV)

# 4.3 Kitaev: pairing breaks U(1) but keeps parity, so grade by FermionParity alone
Vp = Vect[FermionParity](0 => 1, 1 => 1)
cp, cdp = matrixunit(Vp, FermionParity(0), FermionParity(1)),
    matrixunit(Vp, FermionParity(1), FermionParity(0))
kitaev(N; t = 1.0, Δ = 0.5) = opsum(
    FiniteChain(Vp, N),
    (-t * (couple(cdp[i], cp[i + 1]) + couple(cdp[i + 1], cp[i])) for i in 1:(N - 1)),
    (Δ * (couple(cdp[i], cdp[i + 1]) + couple(cp[i + 1], cp[i])) for i in 1:(N - 1)),
)
report("4.3 Kitaev chain (parity only)", kitaev(6))

# 4.4 non-abelian *and* fermionic: U(1) charge x SU(2) spin x parity.
# `matrixunit` is refused per *sector*, not per space: it reaches the dim-1 vacuum/double sectors
# (so `U n↑n↓` is a matrixunit) but not the spin-½ doublet, which has quantum dimension 2 — so any
# operator touching it, the hopping included, goes through `project`. The physical Hubbard bond is
# in `benchmark/ShowcaseModels.jl` (`hubbard_su2`, checked against the spin-orbital encoding); the
# random block here stands in for it to exercise the machinery on a *generic* dense block.
const Hub = ProductSector{Tuple{U1Irrep, SU2Irrep, FermionParity}}
Vh = Vect[Hub]((0, 0, 0) => 1, (1, 1 // 2, 1) => 1, (2, 0, 0) => 1)
hubblock = project(randn(ComplexF64, Vh ⊗ Vh ← Vh ⊗ Vh))   # stand-in for -t(c†c + h.c.) + U n↑n↓
report(
    "4.4 SU(2) Hubbard bond (projected)",
    opsum(FiniteChain(Vh, 4), hubblock[i] for i in 1:3),
)

# ── Tier 5 · the lattice itself ────────────────────────────────────────────────────────────────
Sa, Sb = spin(V½), spin(V1)
alt = FiniteChain([V½, V1, V½, V1])                # the case FiniteChain(V, N) cannot express
report(
    "5.1 alternating spin-½ / spin-1",
    opsum(alt, dot((isodd(i) ? Sa : Sb)[i], (isodd(i + 1) ? Sa : Sb)[i + 1]) for i in 1:3),
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
    opsum(FiniteChain(V½, 12), dot(S[i], S[j]) for (i, j) in cyl_bonds(4, 3)),
)

# ── Tier 7 · long range and truncation ─────────────────────────────────────────────────────────
powerlaw(N; α = 2.0) = opsum(
    FiniteChain(V½, N),
    abs(m - n)^(-α) * dot(S[n], S[m]) for n in 1:(N - 1) for m in (n + 1):N
)
report("7.1 power law α=2, N=10", powerlaw(10))
let H6 = powerlaw(6), oracle = instantiate(H6), l6 = H6.lattice
    for k in (8, 4, 2)
        Wt, st = irrep_mpo(H6, SVDBondAlgorithm(truncrank(k)))
        err = norm(mpo_tensormap(irrep_mpo_tensors(Wt, st, l6)) - oracle) / norm(oracle)
        println(
            "     truncrank($k): D_dense=", maximum(s -> sum(dim, s), st),
            "  rel.err=", round(err; sigdigits = 3)
        )
    end
end

# ── Tier 8 · infinite chains ───────────────────────────────────────────────────────────────────
# the *same* container and call; only the lattice and the return type change
inf1 = irrep_mpo(opsum(InfiniteChain([V½]), dot(S[1], S[2])))
println(rpad("8.1 infinite Heisenberg (L=1)", 42), " ", inf1)
inf2 = irrep_mpo(
    opsum(InfiniteChain([V½, V½]), 0.6 * dot(S[1], S[2]) + 1.4 * dot(S[2], S[3])),
)
println(rpad("8.2 dimerised, L=2", 42), " ", inf2)
# 8.3 exponential decay: finite and infinite, same operator
hexp = (dot(S[1], S[2]), expterm(dot(S[1], S[2]); decay = 0.4))
println(rpad("8.3 exp decay, FiniteChain", 42), " ", irrep_mpo(opsum(FiniteChain(V½, 8), hexp)))
println(rpad("8.3 exp decay, InfiniteChain", 42), " ", irrep_mpo(opsum(InfiniteChain([V½]), hexp)))

# ── Tier 9 · operators that are not Hamiltonians ───────────────────────────────────────────────
Sops = spin_ops(Vu, up, dn)
lat5 = FiniteChain(Vu, 5)
report("9.1 total magnetisation Σ Sᶻ", opsum(lat5, Sops.Sz[i] for i in 1:5))
report("9.2 correlator Sᶻ₁Sᶻ₄", opsum(lat5, couple(Sops.Sz[1], Sops.Sz[4])))
# 9.3 a *charged* operator: fine for irrep_mpo, refused by the Jordan emission
chg = opsum(FiniteChain(Vf, 4), F.cd[2])
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
report("9.4 string n₁n₂n₃n₄", opsum(FiniteChain(Vf, 4), couple(F.n[1], F.n[2], F.n[3], F.n[4])))

# ── Tier 10 · handing it onward ────────────────────────────────────────────────────────────────
Wj = jordan_mpo_tensors(heisenberg(6))
println(rpad("10.2 Jordan form bond sizes", 42), " ", map(W -> size(W, 4), Wj))
Ti = irrep_mpo_tensors(inf1, InfiniteChain([V½]))
println(
    rpad("10.3 infinite tensors tile", 42), " ",
    space(Ti[1], 1) == space(Ti[end], 4)'
)

println("\nPROGRESSION OK")
