# Handoff: finishing the MPSKit absorption

*Handoff prompt for a fresh Claude Code session started in this repo, on branch `cleanup`. Written
after stages 0–4 of the cleanup landed. The audit that motivated the work, and the three of its
claims that turned out to be wrong, are summarised below — read §4 before trusting anything in the
older notes.*

---

## 1. Objective

Make OpSum.jl a slim, standalone package that **MPSKit.jl depends on** and that fuels its
Hamiltonian construction, replacing the greedy channel assignment plus fixed-point space deduction
in `FiniteMPOHamiltonian(lattice, local_operators)` with exact minimum-bond-dimension compression
that already knows every bond space.

Scope decisions already taken, and **not** open for re-litigation:

| | |
|---|---|
| Front-end | Both. The ITO algebra (`couple`/`dot`/`spin`) is the primary construction path — no re-projection per term. `project` stays first-class for *obtaining* on-site operators once, outside any loop, and for importing dense `K`-site operators. |
| Geometry | **Finite only.** Keep the graph and `BondStrategy` interfaces free of finite-only assumptions so a unit-cell fixed-point sweep can be added later; do not build it. |
| Packaging | Slim standalone package; MPSKit adds it as a dependency. OpSum must **never** depend on MPSKit (test-only is fine). |
| SVD backend | Both truncation semantics kept, behind `SVDBondAlgorithm(trunc; sweep)`. |
| Output seam | `Vector{SparseBlockTensorMap}` in Jordan order, via BlockTensorKit. Done — see §3. |

---

## 2. Where things stand

Six commits on `cleanup`, one per stage, each verified by a full-suite run:

```
4ee4084  Collapse the term algebra onto one append-only column store
46129dc  Emit the compressed MPO in Jordan form as SparseBlockTensorMaps
e559ffc  Update CLAUDE.md for the OnsiteOp refactor
3417849  One sweep skeleton, three bond-basis strategies, no oracle
c9e6ccc  Replace the LocalOp sum type with a concrete OnsiteOp
da01b7d  Hygiene: consolidate exports, run Aqua, dedupe test helpers
49cb483  (upstream) Make the reduced-MPO sweep linear for finite-range models (#20)
```

Current state: **3182 tests pass, 0 broken, 14 files, ~10m50s.** `src/` is 2736 non-comment lines
(from 2471). Dependencies: TensorKit, BlockTensorKit, MatrixAlgebraKit, VectorInterface,
SparseArrays, Dictionaries, LinearAlgebra — all of which MPSKit already has except `Dictionaries`.

What each stage did, in one line:

- **0** — stale comments referencing deleted files; exports consolidated into `src/OpSum.jl`; Aqua
  wired (it found real type piracy, now removed); duplicated test helpers → `test/testutils.jl`.
- **1** — `LocalOp` (a 7-variant `LightSumTypes.@sumtype`, only 3 variants ever constructed) →
  concrete `OnsiteOp{I}`, a letter→coefficient map. The bare identity is the `passthrough` sentinel
  letter, not a separate field, so consumers branch on `ispassthrough`. LightSumTypes dropped.
- **3** — the `Θ(M·N²)` transient-frontier oracle deleted; one sweep skeleton with a pluggable
  `BondStrategy` (`VertexCover`, `IndependentSVD`, `SequentialSVD`); every load-bearing `@assert`
  became a real `throw` via a `@noinline _invariant` helper.
- **4** — `jordan_mpo_tensors`: Jordan-ordered `SparseBlockTensorMap` emission (`jordanmpo.jl`).
  MPSKit wired as a **test-only** dep, verified end to end including `find_groundstate` vs ED.
- **2** — `TermKey`/`TermSum`/`ITOTermTable` collapsed onto an append-only column store carrying the
  lattice. `reduce(+, generator)` went from `N^2.29` to `N^0.95` (10.4 s → 0.006 s at N=8192).

---

## 3. What to build

### Stage 7 — MPSKit / MPSKitModels integration (highest value; do this first)

The emission side is done. `jordan_mpo_tensors(H[, sites][, alg])` returns one
`SparseBlockTensorMap` per site, `W_i : B_{i-1} ⊗ V_i ← V_i ⊗ B_i`, virtual legs `SumSpace`s with one
level `Vect[I](charge => 1)` per bond index, ordered `(start, …, finish)`, identity at `(1,1)` and
`(end,end)`, diagonal unit pass-throughs as `BraidingTensor`s.

Remaining work is **MPSKit-side**, in the worktree `/mnt/home/ldevos/Projects/MPSKit.jl/opsum`
(currently an empty branch identical to `main`):

1. A `FiniteMPOHamiltonian` constructor taking OpSum's emission. Bulk tensors already pass through
   `JordanMPOTensor(::SparseBlockTensorMap)` unmodified. **The boundary tensors do not** — that
   constructor asserts both diagonal corners are identities, but at site 1 (single row) the
   `(end,end)` corner *is* the `(1,end)` D-block slot, and symmetrically at site N. Route the
   boundaries through the same `undef` + `setindex!` path the constructor itself uses.
2. **The test env tracks MPSKit `main`, not the registered release.** These are different code even
   though both report version `0.13.13`: `JordanMPOTensor` was reworked after the tag, so registered
   has fields `V,A,B,C,D` while main has `tensors,scalars`. `test/Project.toml` pins
   `MPSKit = {url = "https://github.com/QuantumKitHub/MPSKit.jl", rev = "main"}` — a url+rev source
   rather than a local path, so it still resolves on CI and on any other machine.

   Two consequences. First, `rev = "main"` is a **moving target**: a resolve after main changes can
   pull code this emission has not been tested against, and the `MPSKit = "0.13.13"` compat entry
   will block main once it bumps to 0.14. Pin `rev` to a SHA if that becomes annoying. Second,
   **Pkg's clone cache can be stale** — if `fieldnames(MPSKit.JordanMPOTensor)` comes back as
   `(:V, :A, :B, :C, :D)` you are on old code; force a fetch of the clone under
   `~/.julia/clones/` and `Pkg.update("MPSKit")`. The seam test passes against main as of
   `1c817907`, including DMRG vs ED, with no test changes needed — it only uses API-level entry
   points (`JordanMPOTensor`, `jordanmpotensortype`), not the struct fields.
3. MPSKitModels adapter: `LocalOperator` is `(Vector{MPOTensor}, Vector{LatticePoint})`. Recombine to
   a dense `K`-site `TensorMap`, `project` it **once per distinct operator** (cache it), then place
   with the ITO algebra. This keeps `@mpoham`'s surface exactly and bypasses `instantiate_operator`'s
   swap-gate + SVD-truncation canonicalisation (`mpo.jl:449-497`) entirely.
4. Unrelated bug found during the audit, worth reporting upstream regardless: **`remove_orphans!` is
   a silent no-op for `MPOHamiltonian`** — `SparseMPO` requires `O <: SparseBlockTensorMap` but
   `MPOHamiltonian`'s element is `JordanMPOTensor`, so `toolbox.jl:328` prunes nothing.

### Stage 5 — API ergonomics

Stage 2 already delivered the lattice (`onlattice`/`lattice`, so `irrep_mpo(H)` takes one argument
and the `(H, sites)` tuple threading is gone) and `≈` on term sums. Still open:

- **Out-of-order `couple` for `UniqueFusion` sectors** (`Trivial`, `U₁`, `ℤₙ`, `FermionNumber`,
  products). This is the most bug-prone thing left in the user-facing API: every fermionic model
  hand-supplies the anticommutation sign, with the same warning comment duplicated in
  `examples/fermions.jl` and `benchmark/ShowcaseModels.jl`, and `examples/fermions.jl` carries a
  `sign` kwarg *specifically so the wrong sign can be demonstrated*. Swapping two uncoupled legs
  costs an R-symbol phase (±1 for fermionic sectors) and the caterpillar inner lines are forced, so
  `couple(c[j], cd[i])` with `i < j` can sort and insert the sign automatically. **Keep the throw for
  genuinely non-abelian out-of-order coupling** — that needs F-moves and is out of scope.
- **Operator builders.** `fermion_ops` lives only in `benchmark/ShowcaseModels.jl`, and the
  `Sp`/`Sm`/`Sz`-from-`matrixunit` block is re-derived in nine files. Promote U(1) spin and fermion
  builders into `src/` alongside `spin(V)`.
- **Memoise `spin(V)` and `matrixunit`.** `spin` recomputes its normalisation on every call and
  `matrixunit` runs a full projection; both are documented as "hoist it out of the loop" rather than
  fixed.
- **`hc(term)`** for the hermitian-conjugate partner. Not implemented.
- **Promote `islossless` and `mpo_tensormap` to public API.** Both are still hand-rolled: `islossless`
  in `test/testutils.jl` and six example sites, `mpo_tensormap`'s `ncon` network copy-pasted into
  `examples/common.jl:134` and `docs/src/operators.md:336`.

### Stage 6 remnants — test coverage

Stage 3 retargeted `test_irrep_graph.jl` off the deleted oracle (round-trip + dense oracle + pinned
sector multisets), and **all the §2.2 collision cases survive**. Still open from the audit:

- Property tests over random small term sums (`Trivial`/`U1Irrep`/`SU2Irrep`/`FermionNumber`,
  `K ∈ 0:4`) asserting losslessness. This is what covers §2.1's running-charge component, which
  differential testing against the old oracle provably could not.
- `K ≥ 4` in the suite (only exercised by `examples/multibody.jl` today).
- `trunctol` at API level (imported but unused in `test_irrep_mpo_svd.jl`).
- Truncation *error quality* — currently only `!(Mtrunc ≈ Mexact)`, never that the error is small or
  decreases with rank.

### Known issue: suite time

`test_jordan_mpo.jl` takes 640 s and sets the parallel wall clock, taking the suite from 5m23s to
10m49s. The cost is DMRG (~3m20s) plus two dense contractions (~2m45s), mostly compilation. Dropping
the `find_groundstate` testset is the single biggest saving, at the cost of the only check that
MPSKit's *algorithms* run on this MPO. Deliberate call, not yet made.

---

## 4. Things that are true and non-obvious

Read these before changing the sweep or the term store.

- **`research/persistent-graph-mpo.md` §2.2 is the load-bearing invariant.** The pending↔started
  suffix-class collision is the thing a change to `irrepgraph.jl` is most likely to break silently.
  `test/test_irrep_graph.jl` guards it in all three sectors.
- **The Jordan finish channel must be *forced*, not read off the cover.** Two vertices can carry
  "already finished" — the exhausted trivial-charge right vertex, and the degree-one left vertex
  hanging off it. When they form an isolated matched pair, König covers the *left* one, whose index
  emits the term's last letter × coefficient rather than `1·id`, so it cannot be the `(end,end)`
  corner. `_force_finish!` handles this, on the Jordan path only. Empirically it costs nothing:
  identical per-bond profiles on all eleven showcase models and on Heisenberg/Hubbard/power-law at
  N up to 64.
- **Jordan padding costs +2 over the whole chain, independent of N**, and never moves the maximum
  bond dimension. Heisenberg N=6: `[2,3,3,3,2,1]` → `[3,3,3,3,3,1]`, the textbook uniform 3.
- **`dot`'s apparent argument-order asymmetry is not reachable.** `couple(…; to = unit(I))` only
  admits mutually dual charges, and dual sectors have *equal* quantum dimension in any unitary fusion
  category, so `-√dim(c_a)` and `-√dim(c_b)` always agree. Requiring equal charges — the obvious
  "fix" — would break the legitimate, tested `dot(raise[1], lower[2])` under U(1).
- **`SequentialSVD` with an aggressive `truncrank` can collapse a non-abelian bond to dimension
  zero** (4-site Heisenberg, `truncrank(1)`: the rank-1 bond keeps the charge-0 channel and cannot
  carry the spin-1 channel the next bond needs). Jordan emission refuses this with a clear error.
  This is a footgun, not merely a different tradeoff — worth a thought before advertising it.
- **`_prefix_ids` needs no charge component, `_suffix_ids` does.** A prefix's factor list determines
  the running charge at every idle site to its left; a suffix's idle sites *before* its first
  remaining factor carry the charge the prefix accumulated. That asymmetry is why §2.1's signature is
  a pair.
- **Invariant checks are real `throw`s, never `@assert`.** They guard silently wrong *output*, and
  `@assert` is strippable. Go through `@noinline _invariant`.
- **The sweep stays scalar.** Non-abelian structure enters only via what makes a bond state distinct
  (`ITOKey.bond`) and later at tensor assembly. Do not introduce quantum dimensions or F-symbols into
  the sweep.
- **`research/port-handoff.md` is historical.** It describes the original ITensor port's scope, not
  the code as it now stands.

---

## 5. Verification

```bash
cd /mnt/home/ldevos/Projects/OpSum.jl/cleanup
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using Pkg; Pkg.test()'          # baseline: 3182 pass, 0 broken, 14 files
```

Run the suite in the background and poll — it takes ~11 minutes, and `Pkg.test` buffers its output
until the end. A much faster inner loop:

```bash
julia --project -e 'using Pkg; Pkg.test(test_args=["test_irrep_mpo","test_irrep_term"])'
```

Bond dimensions are deterministic and machine-independent; several tests pin exact per-sector
multisets (`test_irrep_mpo.jl`, `test_showcase_models.jl`, `test_jordan_mpo.jl`). If one moves,
that is a real change — state the old and new values and justify it, never weaken the assertion.

Scaling guards (timings are wall-clock on a shared machine; treat the fitted exponent as the
headline, good to about one decimal):

```bash
julia --project=benchmark benchmark/run.jl --sweep ci
```

Format everything you touch with Runic (not a project dependency):

```bash
julia --project=/tmp/runic-opsum -e 'using Pkg; Pkg.add("Runic")'
julia --project=/tmp/runic-opsum -e 'using Runic; for f in ARGS; Runic.format_file(f, f; inplace=true); end' <files>
```
