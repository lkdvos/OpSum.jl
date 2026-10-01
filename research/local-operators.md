# Unplaced multi-site operators and a lattice-carrying `OperatorSum` — design note

Decided 2026-10-01, not yet implemented.
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
  graded sign, so `B[i, i + 2]` of a projected `c†c` block should be the Jordan–Wigner-correct hop.
  **Unverified** — needs a dense `instantiate` test before it is documented.
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
2. **Unplaced `couple` / `dot` / `couple_channels`, passthrough slots.** Tests: `couple(a, b)[sites...]` ≡
   placed `couple(a[i], b[j])` across abelian, fermionic and SU(2) sectors; forced-channel errors unchanged.
3. **`OperatorSum`.** Absorb `MixedSum`, remove the `(h, lat)` forms and `adjoint(h, lat)`, add `H'`, move
   validation to insertion; migrate examples, benchmarks and `ShowcaseModels.jl` builders (which return
   `(h, lat)` today and would return an `OperatorSum`). Watch the Julia 1.10 stale-`using` trap when
   removing exports.

## 8. Open

* Do placed `couple`/`dot` survive step 2, or become sugar for unplaced-then-place?
* Does `OperatorSum` canonicalise eagerly on insertion, or keep today's normalise-on-observation?
* `H'` on a container holding channels: throw, or adjoint the channel representative?
