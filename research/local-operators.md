# Unplaced multi-site operators and a lattice-carrying `OperatorSum` — design note

Decided 2026-10-01. Step 1 (§3: `LocalOperator`, `project(h)`, placement) and step 2 (§4: unplaced
`couple`/`dot`/`couple_channels`, and the pass-through slots of §3) implemented 2026-10-01 on the
`local-operators` branch; step 3 (§5, `OperatorSum`) implemented 2026-10-01 on the same branch.
Builds on `interface-review.md` (read §1, §10–12 first: this note reverses part of §12, deliberately, and
§1 is the failure mode the new container must not repeat).

---

## 1. Motivation

Two gaps, found while reading `project`.

**The type grid has a hole.**

| | unplaced | placed |
|---|---|---|
| **1 site** | `SiteOperator` | `Terms` (via `O[i]`) |
| **K sites** | — | `Terms` |

A `Term` is `(sites, keys, coeff)`, so there is no way to hold a multi-site operator before deciding where
it acts. `project(h, sites)` therefore has to take the sites, and `couple`/`dot` only accept placed
operands.

**Every model re-projects the same tensor once per bond.**

```julia
opsum(project(h_bond, [i, i + 1]) for i in 1:(N - 1))     # examples/spin_chains.jl:138
```

Also `spin_chains.jl:176`, `fermions.jl:323`, `benchmark/ShowcaseModels.jl:188`,
`research/progression.jl:80,140`. Each call is a full projection plus its dense faithfulness check, `N - 1`
times for one shape. `adjoint` already memoises per term shape; the shape is just not a user-facing value.

## 2. Decisions

| question | decision |
|---|---|
| new type for unplaced K-site operators | **yes**, `LocalOperator{I}`; `SiteOperator` stays separate |
| terms on a subset of `1:K` | **no**; every term has exactly `K` keys |
| scalars / identity factors in unplaced arithmetic | a `passthrough` letter occupies the slot |
| `couple` / `dot` on unplaced operands | **yes** — keeps placement out of the coupling |
| spaces on `LocalOperator` | **no**, not now (§6) |
| where spaces enter | a lattice-carrying container, `OperatorSum(lat)` |
| `H += x` | **allowed**; quadratic when folded, `opsum!` is the linear route |
| `irrep_mpo(h, lat)` and the other `(h, lat)` forms | **removed**; everything takes an `OperatorSum` |
| block-level lazy `B'` | **deferred** (§6); container-level `H'` only |

## 3. `LocalOperator{I}`

**Representation.** A `Terms{I}` on relative sites `1:K`, plus `K`. That reuses `canonicalize!`, `≈`,
`==`/`hash` and `show` unchanged. Invariant: every term has `sites == 1:K`.

**Why not merge with `SiteOperator`.** A one-site term's key is `(letter, letter.c, 1)`, so K = 1 is
isomorphic to a `SiteOperator`. But `SiteOperator` is the entry type of every bond matrix in the sweep
(`Dictionary{CartesianIndex{2}, SiteOperator}`) and must stay a flat letter → coefficient map. Provide a
conversion, keep the types apart; "site" means one site, "local" means K.

**Passthrough slots.** `dot(S, S) + 1/4` and `couple(Sz + 1/2, Sz)` produce terms that are not active on
every slot. Rather than relax the invariant or materialise identity letters (which needs spaces: whether a
trivial-charge letter *is* the identity depends on the site's space), the `passthrough` sentinel occupies
the slot. The running bond charge passes through unchanged. Placement drops passthrough keys, so placed
`Term`s and everything downstream are untouched; a term that is passthrough in every slot places as a
`K = 0` scalar term, exactly as `SiteOperator` placement does today.

**Implemented (step 2).** The pass-through key's `bond` is the running charge *before* the slot, which is
only `unit(I)` when nothing charged precedes it: in `couple(Sp, Sz + α, Sm)` the middle key is
`(passthrough, U₁(1), 1)`. `_padkey(I)` (irreptermtable.jl) has `bond = unit(I)` and would be wrong
there; it is not used on this path — `_couple_terms` writes `target(run, passthrough) = run` itself, so
no special case exists. `test/test_local_couple.jl` ("pass-through slot inside a charged caterpillar")
checks the key and the dense placement for U(1), fermions and SU(2). The sources of pass-through slots
are `B + α` (= `B + α · one(B)`, the all-pass-through term), a `SiteOperator` with a scalar part as a
`couple` operand (`SiteOperator ± Number` was added for `Sz + 1/2`), and `LocalOperator(::SiteOperator)`.
`project(h)` never produces one — every slot of a projected term is a letter — so `project(S·S + ¼)` and
`dot(S, S) + 1/4` are the same operator spelled differently, and place to *different* bags.

**Placement.**

```julia
B = project(h_bond)              # ::LocalOperator, K = 2
B[i]                             # contiguous: sites i:i+K-1
B[i, i + 2]                      # explicit; strictly increasing, length K
project(h, sites) == project(h)[sites...]     # kept as a one-line convenience
```

* Strictly increasing only, so the caterpillar tree and bond charges carry over for *any* symmetry.
  Non-monotone placement is a sector-only reordering (R-symbols abelian, F-moves non-abelian) and is
  deferred (§6).
* Sites in a gap become pass-throughs. For fermionic sectors the bond charge crossing the gap picks up the
  graded sign, so `B[i, i + 2]` of a projected `c†c` block is the Jordan–Wigner-correct hop.
  **Verified** (`test/test_local_operator.jl`, "gap placement is exact"): `B[1, 3]` of the projected
  nearest-neighbour hop equals `couple(cd[1], c[3]) + h.c.` as a term bag and densely under
  `instantiate`; the compressed MPO (`irrep_mpo_tensors`, pass-through at the gap site) contracts back to
  it; and two oracles that bypass the term algebra hold — the triangle `B[1] + B[2] + B[1, 3]` has zero
  flux (one-particle spectrum `τ·{2, -1, -1}`, not the frustrated `τ·{-2, 1, 1}`), and its two-particle
  spectrum is exactly the pair sums of the one-particle one, which is where the string across the
  occupied gap site acts. Same checks pass for SU(2) `S·S` (`(S_tot² - 9/4)/2` on the triangle) and U(1).
  One finding along the way: `couple(cd[1], c[2])` materialised as `-|10⟩⟨01|` in the product basis —
  uniform over all bonds, so invisible on bipartite graphs, physical on odd loops. Ruled a bug and
  fixed separately (#36): `_couple_terms` multiplies a coupled pair by `-1` when both operands are
  fermion-odd (the graded tensor product), so `couple(cd[i], c[j])` is the physical `c†ᵢcⱼ`; the
  letters are unchanged, and `test/test_fermion_signs.jl` pins it against dense Jordan–Wigner
  matrices. Unplaced `couple`/`dot` lower onto `_couple_terms`, so they inherit it; any new route
  that combines operands into one term must apply the same rule.
* `Terms` indexing (`ts[i]` = i-th canonical term, `irrepalgebra.jl:206`) is unaffected; that clash is why
  placement lives on the new type rather than on `Terms`.

**Arithmetic.** `+`/`-` between operators of equal `K` (error otherwise), scalar `*`/`/`, `zero`, `copy`.

## 4. Unplaced `couple` and `dot`

* `couple(a, b...; to, via)` with every operand a `SiteOperator` or `LocalOperator` returns a
  `LocalOperator` with `K = Σ K_i`: slots are concatenated, the second operand's relative sites shifted by
  `K_a`. The forced-channel fold and `couple_channels` from #34 carry over unchanged — they only look at
  charges.
* Slot order *is* written order, so the unplaced path never reorders: no R-symbol, no `_canreorder` gate.
  `dot(S, S)` keeps its per-letter `-√dim(c)` factor and needs no braiding.
* Mixing placed and unplaced operands is an error.
* Placed `couple`/`dot` stay for now. Whether they become redundant once `couple(cd, c)[i, j]` covers every
  hop is an open question (§8).

**Implemented (step 2)** in `src/operators/algebra/localoperator.jl`, by lowering: each operand is placed
on consecutive *slots* (`_slotterms`, keeping pass-through letters as keys; `_slotoperands`), the placed
`couple`/`couple_channels`/`dot` runs unchanged, and the result is wrapped with `K = Σ K_i`. So there is one
implementation of the channel arithmetic and one set of error messages (`test_local_couple.jl` compares
the strings). Two consequences worth knowing:

* Every operand after the first must be **single-slot**: the caterpillar extends by one letter, and
  fusing a `K ≥ 2` block onto it is `via`. The first operand may have any `K`, so
  `couple(couple(S, S; to = 1), S)` nests as before. The error names the slot count.
* `couple`, `couple_channels` and `dot` gained a catch-all method over
  `Union{Terms, Term, SiteOperator, LocalOperator}` that throws the mixing error; an all-`Term` call
  still raises a `MethodError`, as it did. `nsites(::SiteOperator) = 1` exists.

## 5. `OperatorSum(lat)`

The one place spaces are confronted. Note the name: the module is `OpSum`, so a type of the same name is
unusable after `using OpSum`.

```julia
H = OperatorSum(FiniteChain(V, N))           # or InfiniteChain([Va, Vb]); an iterable of spaces → FiniteChain
H += B[i]                                    # copies: folding it over N terms is Θ(N·M)
opsum!(H, (B[i] for i in 1:(N - 1)))         # the linear route; docs use this
H = opsum(lat, terms...)                     # one pass, as today's opsum
H += expterm(t; decay = λ)                   # channels too: absorbs MixedSum
H = H + H'                                   # needs total charge unit(I) per term
irrep_mpo(H[, alg]); islossless(H[, alg]); jordan_mpo_tensors(H[, alg]); instantiate(H)
```

* **It must cover all four cells of `interface-review.md` §1.** `TermSum` failed there: no infinite chains,
  no `expterm`, and `H'` missing exactly where `T + T'` is most wanted. Here a finite or infinite lattice,
  with or without channels, is the same container, and `irrep_mpo(H)` has one signature.
* **Validation on insertion.** `_checklattice` (letters, sector type, site range) runs when terms enter, so
  errors point at the line that added the term rather than at `irrep_mpo`.
* **What needs spaces, and so lives here:** letter validation, `adjoint`, `instantiate`, and — later — the
  W8 identity stripping (`S₁·S₂ + ¼` returning two two-site terms). R- and F-symbols do *not* need spaces;
  reordering stays where it is (`_couple_terms`, eager).
* **Unchanged:** the sweep, `ITOTermTable(ts, N)` (the container hands it `length(lat)`), and
  `mpo_terms(Ws, secs)`, which still returns a latticeless `Terms` — `islossless` compares it against the
  container's terms.
* **Removed:** every `(h, lat)` form — `irrep_mpo`, `islossless`, `jordan_mpo_tensors`, `instantiate`,
  `adjoint(h, lat)` — and `MixedSum`. `opsum(::AbstractLattice, …)` currently throws `_nolattice()`
  (`irrepalgebra.jl:377`); that method is repurposed, and the no-lattice `opsum(args...)` returning `Terms`
  stays as the latticeless accumulator.

The four cells, redone:

```julia
V = SU2Space(1 // 2 => 1); S = spin(V); b = dot(S, S)       # b::LocalOperator, never placed

H = opsum(FiniteChain(V, 8), (b[i] for i in 1:7))                                  # A
H = opsum(InfiniteChain([V]), b[1])                                                # B
H = opsum(FiniteChain(V, 8), b[1], expterm(b[1]; decay = 0.4))                     # C
H = opsum(InfiniteChain([V]), b[1], expterm(b[1]; decay = 0.4))                    # D
mpo = irrep_mpo(H)
```

## 6. Deferred

* **Block-level lazy adjoint `B'`.** Designed, not scheduled. A single flag cannot represent `B + B'`, so it
  is a second *pending* bag on `Terms` (and hence `LocalOperator`): `'` swaps the bags (exact, since
  `(X + Y†)† = X† + Y`), scalars conjugate on the pending side, canonicalisation is per bag, the total
  charge check is eager, and `OperatorSum` resolves pending terms on insertion through the existing
  per-shape `adjoint` memo. `couple`/`dot`/`expterm` must throw on pending operands — a half-adjointed
  coupled term has no representation. Removing the `(h, lat)` forms is what keeps pending terms from ever
  reaching the sweep.
* **Spaces on `LocalOperator`** (`interface-review.md` §10, B″, in multi-site form). Would give `B'`
  eagerly, `instantiate(B)` and earlier validation, without breaking #32's efficiency argument (a
  `LocalOperator` never reaches the compression). Held back on ergonomics, not correctness.
* **Terms on a subset of `1:K`.** Passthrough slots cover the cases met so far.
* **Non-monotone placement** `B[j, i]`: sector-only, resolvable at placement.
* **Unplaced `expterm`** (a `LocalOperator` entry block).

## 7. Implementation order

One PR per step, each independently testable.

1. **`LocalOperator`, `project(h)`, placement.** Migrate the per-bond `project` loops (§1). Tests: placement
   ≡ `project(h, sites)`; gap placement against dense `instantiate`, including a fermionic hop.
   **Done** — `src/operators/algebra/localoperator.jl`, `test/test_local_operator.jl`; all five loops in §1
   migrated and checked to give identical term bags. `LocalOperator(::SiteOperator)` refuses a
   `passthrough` letter until step 2 defines passthrough slots; placement does not yet drop passthrough
   keys (nothing produces them yet).
2. **Unplaced `couple` / `dot` / `couple_channels`, passthrough slots.** Tests: `couple(a, b)[sites...]` ≡
   placed `couple(a[i], b[j])` across abelian, fermionic and SU(2) sectors; forced-channel errors unchanged.
   **Done** — `localoperator.jl` ("Unplaced coupling" section, pass-through handling in `getindex`),
   `siteoperator.jl` (`SiteOperator ± Number`), `test/test_local_couple.jl`. Structural `==` (not just `≈`)
   against the placed form for every sector, 2–4 operands, nested and variadic, contiguous and gapped;
   `couple_channels` unplaced == placed; pass-through slots placed and densely checked, including inside a
   charged caterpillar; lossless chains from unplaced blocks; error strings identical to the placed ones.
   Examples were *not* migrated (step 3 rewrites them); `docs/src/operators.md` has a "Building blocks
   unplaced" section.

   What step 3 must know:
   * A `LocalOperator`'s **raw terms may contain pass-through keys**; placed `Terms` never do. Anything in
     step 3 that reads a `LocalOperator`'s terms directly rather than through placement — `instantiate(B)`
     with spaces, validation of unplaced operands, a lazy `B'` — has to treat `ispassthrough(k.op)` as the
     identity on that slot, with `k.bond` the running charge (not necessarily `unit(I)`). `_checklattice`
     runs on placed terms and is unaffected.
   * `OperatorSum` insertion through `B[i]` sees ordinary `Terms`, so nothing in the container needs to
     know about slots. If it accepts an unplaced `LocalOperator` directly (`H += B` with an implied site),
     that is a new decision.
   * The W8 identity stripping in §5 (`S₁·S₂ + ¼` → two two-site terms) only arises for `project`ed blocks;
     `dot(S, S) + 1/4` already places as a two-site term plus a `K = 0` constant.
   * The variadic error messages mention channels, not sites, which is what makes them identical between
     the two forms; keep it that way if they are reworded.
3. **`OperatorSum`.** **Done** — see "Implemented (step 3)" below. Absorb `MixedSum`, remove the `(h, lat)` forms and `adjoint(h, lat)`, add `H'`, move
   validation to insertion; migrate examples, benchmarks and `ShowcaseModels.jl` builders (which return
   `(h, lat)` today and would return an `OperatorSum`). Watch the Julia 1.10 stale-`using` trap when
   removing exports.

**Implemented (step 3).** `src/operators/operatorsum.jl` (struct, `opsum!`, `opsum(lat, …)`, insertion
checks, arithmetic, `H'`, `instantiate(H)`, `chain_terms(H)`), `test/test_operator_sum.jl`. As decided:
`OperatorSum{I, L}` holds the lattice, a `Terms{I}` and an `ExpSum{I}`; `MixedSum` and every `(h, lat)`
form are gone (`irrep_mpo(::Terms/Term/ExpSum, …)` throws an `ArgumentError` naming
`irrep_mpo(opsum(lat, h))`; the rest are plain `MethodError`s); canonicalisation stays lazy.
`irrep_mpo_tensors(mpo, lat)` is the one remaining `(x, lat)` form — the MPO does not carry its
lattice, so it could read it off the MPO later, which would also make `mpo_tensormap ∘
irrep_mpo_tensors` argument-free.

Choices the note left open, and what was done:

* **Where each check runs.** On insertion: sector type, letters against the site's space, finite-chain
  site range (`_checklattice`), and on an `InfiniteChain` the per-term generating-set rules — charge
  neutrality and no `K = 0` term — for terms and channels, plus the letters of a channel's
  representative (translation by `L` preserves the space). At `irrep_mpo`: only "one representative
  per translation class", which needs the whole set (`unitcell_terms`). A channel's letters are *not*
  checked on a finite chain: every translate sits on a different site, so there is no single space to
  check them against. A failed insertion leaves the container unchanged (everything is collected and
  checked before anything is appended).
* **`unitcell_terms(::Terms, L)` now canonicalises its input first**, so a term written twice is one
  term with a summed coefficient rather than a spurious "translates of each other" error. Needed for
  `H + H'` on an infinite chain, where on-site and self-adjoint pieces coincide; it is also what a
  finite chain already did.
* **`islossless` on an infinite chain** is the `mpo_terms_window` sandwich over the same window the
  sweep uses (soundness everywhere, completeness further than the interaction range from the right
  edge), since there is no exact equality to test. On a finite chain it is `mpo_terms(mpo) ≈
  chain_terms(H)`, so channels are covered too.
* **`length`/iteration** of an `OperatorSum` run over the finite-range terms; channels are
  `H.channels`; `isempty` is true only with neither. `≈`/`==` also require equal lattices.
* **`H'` with channels** throws an `ArgumentError` (§8, third question: throw, for now).
* **`H += x`** is supported for `Term`, `Terms`, `ExpSum`; `H + H2` requires equal lattices. An unplaced
  `SiteOperator`/`LocalOperator` (`opsum!(H, B)`) is an `ArgumentError` saying to place it first —
  accepting `H += B` with an implied site stays a separate decision, as step 2 noted.
* **`couple`/`dot`/`couple_channels` on an `OperatorSum`** throw an `ArgumentError` when it is the
  first or second operand; a third or later operand falls through to the generic `MethodError`.
* **`jordan_mpo_tensors`** keeps its finite-only, channel-free scope; the infinite and channel cases are
  `ArgumentError`s.
* **Bug found while migrating:** `project` verified its output with `instantiate(out, Vs)` on a bare
  `Terms`; it now calls the internal `_instantiate_terms(out, Vs)`. `instantiate(::Terms, ...)` is not
  public any more, and tests that need a dense matrix of a latticeless bag use
  `instantiate(opsum(sites, ts))`.

## 8. Open

* Do placed `couple`/`dot` survive step 2, or become sugar for unplaced-then-place? (Step 2 went the other
  way round — unplaced is implemented *on* placed — so removing the placed form means moving the
  reordering/insertion logic of `_couple_terms` to a placement step, which is exactly the deferred
  non-monotone placement of §6.)
* ~~Does `OperatorSum` canonicalise eagerly on insertion, or keep today's normalise-on-observation?~~
  Lazy (decided, implemented).
* `H'` on a container holding channels: throws for now (implemented); adjointing the channel
  representative is open.
* `irrep_mpo_tensors(mpo, lat)` could read the lattice off the MPO.
