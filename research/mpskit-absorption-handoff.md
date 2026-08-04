# Handoff: finishing the MPSKit absorption

*Handoff prompt for a fresh Claude Code session started in this repo, on branch `cleanup`. Written
after stages 0–7 of the cleanup landed. The audit that motivated the work, and the claims of it that
turned out to be wrong, are summarised below — read §4 before trusting anything in the older notes.*

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
| Output seam | `Vector{SparseBlockTensorMap}` in Jordan order, via BlockTensorKit. Done, both sides — see §2. |

---

## 2. Where things stand

Nine commits on `cleanup`, plus one on the MPSKit side:

```
(OpSum, branch `cleanup`)
  <stage 6>  Cover the term space with random-term-sum property tests
  dfe1044    Order-free coupling, hc, operator builders, public verification helpers
  4dda971    Target MPSKit main rather than the registered release
  b57bffa    Add a handoff note for the remaining MPSKit absorption work
  4ee4084    Collapse the term algebra onto one append-only column store
  46129dc    Emit the compressed MPO in Jordan form as SparseBlockTensorMaps
  e559ffc    Update CLAUDE.md for the OnsiteOp refactor
  3417849    One sweep skeleton, three bond-basis strategies, no oracle
  c9e6ccc    Replace the LocalOp sum type with a concrete OnsiteOp
  da01b7d    Hygiene: consolidate exports, run Aqua, dedupe test helpers
  49cb483    (upstream) Make the reduced-MPO sweep linear for finite-range models (#20)

(MPSKit, worktree /mnt/home/ldevos/Projects/MPSKit.jl/opsum, branch `opsum`)
  8ec7f9b7   Accept Jordan-ordered SparseBlockTensorMaps, boundaries included
```

Dependencies: TensorKit, BlockTensorKit, MatrixAlgebraKit, VectorInterface, SparseArrays,
Dictionaries, LinearAlgebra — all of which MPSKit already has except `Dictionaries`.

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
- **7** — MPSKit-side: `JordanMPOTensor(::SparseBlockTensorMap)` now accepts *boundary* tensors, and
  `FiniteMPOHamiltonian(::AbstractVector{<:SparseBlockTensorMap})` takes the whole emission. See §3.1.
- **5** — out-of-order `couple` for abelian sectors (the braiding phase is inserted, so the fermionic
  sign is no longer the caller's job), `hc`, `spin_ops`/`fermion_ops`, memoised `spin`/`matrixunit`,
  and `islossless`/`mpo_tensormap` promoted to public API.
- **6** — property tests over random term sums (`test/test_irrep_properties.jl`), `K = 4` coverage,
  `trunctol` at API level, and truncation *error quality* rather than merely "the operator changed".

---

## 3. What is left

### 3.1 Stage 7 remainder

Done: the MPSKit-side seam. `8ec7f9b7` in the `opsum` worktree fixes
`JordanMPOTensor(::SparseBlockTensorMap)` — it asserted both diagonal corners were identities, which
a boundary tensor cannot satisfy (site 1 has one row, so its `(end, end)` corner *is* the `(1, end)`
D-block slot, and symmetrically at site N) — by checking exactly the corners the `undef` constructor
installed, which already got the shape logic right. On top of that it adds
`FiniteMPOHamiltonian(::AbstractVector{<:SparseBlockTensorMap{TT,E,S,2,2}})`, which is the whole-chain
entry point. Verified by the full `test/operators/mpohamiltonian.jl` (all testsets, including a new one
that round-trips a lattice-built Hamiltonian through `SparseBlockTensorMap` and back), and by DMRG on
OpSum's emission.

Still open:

1. **Push `8ec7f9b7` and open a PR against MPSKit `main`.** Not done — pushing is an outward-facing
   action. Until it lands, OpSum's `test/test_jordan_mpo.jl` must keep its local `to_jordan` helper
   (the test env tracks `rev = "main"`, so it sees upstream, not the worktree). Once merged, `to_jordan`
   and the paragraph above it collapse to `FiniteMPOHamiltonian(Ws)` and the two `@testset`s that use it
   simplify. That is the only OpSum-side change stage 7 still wants.
2. **MPSKitModels adapter** — *blocked on registration, not on design.* `LocalOperator` is
   `(Vector{MPOTensor}, Vector{LatticePoint})`: recombine to a dense `K`-site `TensorMap`, `project` it
   **once per distinct operator** (cache it), then place with the ITO algebra. That keeps `@mpoham`'s
   surface exactly and bypasses `instantiate_operator`'s swap-gate + SVD-truncation canonicalisation
   (`mpo.jl:449-497`) entirely. The blocker: MPSKitModels would need OpSum as a dependency, and OpSum is
   unregistered, so nothing committed there resolves on CI. Register OpSum first. (There is also no
   reserved MPSKitModels worktree; `/mnt/home/ldevos/Projects/MPSKitModels.jl` has `main`, `kagome`,
   `tensorkittensors`, `testing`.)
3. **Report `remove_orphans!` upstream** — verified still true on `main`, and unrelated to this work.
   `remove_orphans!(mpo::SparseMPO)` (`src/operators/abstractmpo.jl:45`) requires
   `O <: SparseBlockTensorMap`, but `MPOHamiltonian`'s element type is `JordanMPOTensor`, which is an
   `AbstractBlockTensorMap` and not a `SparseBlockTensorMap`. So `MPOHamiltonian` silently takes the
   `remove_orphans!(::AbstractMPO) = mpo` no-op fallback, and `toolbox.jl:328` prunes nothing. Fixing it
   properly needs index-mask slicing on `JordanMPOTensor`, which it does not support — so this is a real
   piece of work, not a one-liner. Filing an issue was left to the human.

### 3.2 Stage 5 remainder

Everything in the original list landed. For the record, what "out-of-order `couple`" turned into:

- `_couple_terms` now *inserts* `b`'s leg at its site-ordered position instead of only appending,
  recomputing the running bond charges from there rightwards and multiplying in one
  `Rsymbol(c_a, c_b, c_a ⊗ c_b)` per leg of `a` that `b` braids past. Gated on `_canreorder(I)` =
  `UniqueFusion` **and** `SymmetricBraiding`.
- `dot` accepts either order for *any* fusion style, because two legs coupling to the unit sector need
  no F-move; it inserts the same scalar R-symbol. This also fixed a latent bug: the old `dot` sorted
  the sites and dropped that phase, which is `-1` for two odd fermionic charges and for two
  half-integer SU(2) charges (`Rsymbol(SU2Irrep(1//2), SU2Irrep(1//2), SU2Irrep(0)) == -1.0`). It was
  only correct because every tested caller had integer operator charges.
- `hc` goes *through* the forward map (materialise the term's `K`-site block, adjoint it, `project` it
  back) rather than symbolically — see the comment in `builders.jl` for why the symbolic route needs
  B-symbols. Memoised per `(letters, tree, spaces)`, so it costs one projection per distinct term
  *shape*, and it inherits `project`'s faithfulness check as a self-test. Requires total charge
  `unit(I)`.

Not attempted, and worth considering only if a caller asks:

- `hc` for **charged** terms. The adjoint lives in the dual charge sector, which means bending the
  `Vect[tot]` leg from codomain to domain as `Vect[dual(tot)]` — a twist/flip, not hard, but it changes
  what the function returns and nothing needs it yet.
- On-site products / powers. Still deferred; `research/onsite-products.md` is the note.

### 3.3 Stage 6 remainder

Everything in the original list landed, in `test/test_irrep_properties.jl`. Two notes:

- The generator projects **random symmetric `K`-site operators** rather than assembling `couple`
  chains. That is what makes it cover the running-charge component of the sweep state: one `project`
  call returns *every* representable (letters, tree) combination over those sites, including the
  caterpillar inner lines a hand-written model rarely varies.
- Runtime is the constraint on how far this goes. The symbolic `islossless` check runs on four sectors
  × three seeds × `K ∈ 0:3`, plus a `K = 4` pass per sector; the *dense* oracle runs on two sectors at
  one size and one seed, because contracting an `N`-site MPO costs a fresh `ncon` specialization per
  `(length, sectortype)` pair.
- `islossless` is **not** usable on an SVD bond basis, and the property tests do not try. That basis is
  a mixture of prefix states, so `mpo_terms` enumerates every path — 3125 terms where the operator has
  165 — and the round-off coefficients of the spurious ones do not cancel to exactly zero. The SVD
  backends are therefore checked two other ways: bond dimensions against the min-vertex-cover bound
  (which they must not exceed), and the operator itself wherever the dense oracle already runs.

### Known issue: suite time

The whole suite is ~9m40s wall clock on 14 parallel workers, set by the two longest files:
`test_irrep_properties.jl` (575 s) and `test_jordan_mpo.jl` (546 s). Adding the property tests cost
almost nothing in wall clock — they run beside the Jordan tests — but they are now the ceiling, so
thinning them is the first lever if it becomes painful (drop a seed, or the `K = 4` pass on the two
abelian sectors).

For `test_jordan_mpo.jl` the cost is DMRG (~3m20s) plus two dense contractions (~2m45s), mostly
compilation. Dropping the `find_groundstate` testset is the single biggest available saving, at the
cost of the only check that MPSKit's *algorithms* run on this MPO. Deliberate call, still not made.

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
- **`JordanMPOTensor`'s corner identities depend on the *shape*.** A boundary tensor has one row or
  one column, and then one diagonal corner coincides with the `(1, end)` on-site slot and must *not*
  be an identity. Raw upper-triangularity `I[1] ≤ I[4]` fails there too — at site `L` the finish
  identity sits at `(end, 1)`. (This is why the corresponding `@assert` in MPSKit was commented out.)
- **`dot`'s R-symbol is not always `+1`.** See §3.2. The older claim in this file — that the apparent
  argument-order asymmetry "is not reachable" because dual sectors have equal quantum dimension — was
  about the `-√dim(c)` factor only, and was read too broadly: the *braiding* phase does not cancel.
- **`SequentialSVD` with an aggressive `truncrank` can collapse a non-abelian bond to dimension
  zero** (4-site Heisenberg, `truncrank(1)`: the rank-1 bond keeps the charge-0 channel and cannot
  carry the spin-1 channel the next bond needs). Jordan emission refuses this with a clear error.
  This is a footgun, not merely a different tradeoff — worth a thought before advertising it.
- **`truncrank(k)` is not `k` reduced bond indices under a non-abelian symmetry.** `svd_trunc`
  truncates globally across sectors with the quantum dimensions counted, so on a spin-½ power-law
  chain whose lossless profile is `[2, 4, 4, 2, 1]`, `truncrank(5)` is still lossy and `truncrank(1:3)`
  produce the *zero* operator (relative error exactly one — the rank cannot hold even one spin-1
  channel). `test_irrep_properties.jl` pins both ends of that. `trunctol` has no such surprise: it is
  stated in the units of the thing being approximated, and its error tracks the tolerance.
- **`_prefix_ids` needs no charge component, `_suffix_ids` does.** A prefix's factor list determines
  the running charge at every idle site to its left; a suffix's idle sites *before* its first
  remaining factor carry the charge the prefix accumulated. That asymmetry is why §2.1's signature is
  a pair.
- **Invariant checks are real `throw`s, never `@assert`.** They guard silently wrong *output*, and
  `@assert` is strippable. Go through `@noinline _invariant`.
- **The sweep stays scalar.** Non-abelian structure enters only via what makes a bond state distinct
  (`ITOKey.bond`) and later at tensor assembly. Do not introduce quantum dimensions or F-symbols into
  the sweep.
- **`spin` and `matrixunit` are memoised**, so "hoist them out of the loop" is no longer advice worth
  giving. Safe because both are pure and `OnsiteOp` has no in-place API — every arithmetic operation
  on it copies. If that ever changes, the caches (`src/utility/memo.jl`) become a correctness bug.
- **`research/port-handoff.md` is historical.** It describes the original ITensor port's scope, not
  the code as it now stands.

---

## 5. Verification

```bash
cd /mnt/home/ldevos/Projects/OpSum.jl/cleanup
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using Pkg; Pkg.test()'
```

Run the suite in the background and poll — it takes ~11 minutes, and `Pkg.test` buffers its output
until the end. A much faster inner loop:

```bash
julia --project -e 'using Pkg; Pkg.test(test_args=["test_irrep_mpo","test_irrep_term"])'
```

Bond dimensions are deterministic and machine-independent; several tests pin exact per-sector
multisets (`test_irrep_mpo.jl`, `test_showcase_models.jl`, `test_jordan_mpo.jl`). If one moves,
that is a real change — state the old and new values and justify it, never weaken the assertion.

The examples are not part of the suite (they are Literate sources for the docs). Run them directly
after touching the public API:

```bash
for f in spin_chains fermions ladders_and_cylinders long_range multibody; do
  julia --project=examples examples/$f.jl || echo "FAILED: $f"
done
```

MPSKit-side changes: the test environment there pulls CUDA/AMDGPU/Plots, so a full
`Pkg.instantiate()` takes tens of minutes. A scratch environment with `MPSKit` dev'd plus TensorKit,
BlockTensorKit, Test, TestExtras, Adapt, Combinatorics, TensorKitTensors runs
`test/operators/mpohamiltonian.jl` directly and is far faster.

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
