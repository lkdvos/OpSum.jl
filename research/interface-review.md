# User-facing interface review (brainstorm, 2026-09-16)

Scope: what a user writes to get a Hamiltonian into OpSum, and how uniform that is across
(finite | infinite) × (plain | exponentially decaying) × (abelian | non-abelian).

## 1. The grid, as it stands

```julia
V = SU2Space(1 // 2 => 1); S = spin(V); h = dot(S[1], S[2])

# A. finite, finite-range
H = opsum(fill(V, 8), (dot(S[i], S[i + 1]) for i in 1:7))
Ws, secs = irrep_mpo(H)

# B. infinite, finite-range              -- no opsum; lattice at the call site
Ws, secs = irrep_mpo(h, InfiniteChain([V]))

# C. finite, exponentially decaying      -- opsum is *not available*
Ws, secs = irrep_mpo(h + expterm(h; decay = 0.4), fill(V, 8))

# D. infinite, exponentially decaying
Ws, secs = irrep_mpo(h + expterm(h; decay = 0.4), InfiniteChain([V]))
```

Four cells, three ways of naming the lattice:

| cell | lattice enters at | operator type | returns |
|---|---|---|---|
| A | `opsum(sites, …)`, **mandatory** | `TermSum` | `(Ws, secs)` tuple |
| B | `irrep_mpo(…, chain)` | `Terms` (or `TermSum`, re-checked) | `InfiniteMPO` |
| C | `irrep_mpo(…, sites)`, `opsum` **impossible** | `MixedSum` | `(Ws, secs)` tuple |
| D | `irrep_mpo(…, chain)` | `MixedSum` | `InfiniteMPO` |

There is no `irrep_mpo(::Terms, ::Vector{<:ElementarySpace})` and no
`opsum(::InfiniteChain, …)` / `opsum(sites, ::MixedSum)`. So the sharpest wart is:
**adding one `expterm` to a finite model changes how you bind the lattice** (A → C), and
**going infinite changes it again** (A → B).

## 2. What that asymmetry costs functionally (not just cosmetically)

`H'` (`adjoint`) is defined on `TermSum` only, and deliberately: it needs the physical spaces.
Cells B, C, D never produce a `TermSum`, so **`T + T'` is unavailable for an infinite hopping
chain** — the one place the shorthand is most wanted. Same story for `instantiate`, `islossless`,
`length(H)`, `canonicalize!`: all keyed on `TermSum`, all conceptually only needing the lattice.
`examples/common.jl:build(name, H::TermSum)` inherits the restriction — the examples' own
reporting harness cannot report an infinite or `MixedSum` model.

## 3. Type zoo the user must be able to name

`SiteOperator`, `Term`, `Terms`, `TermSum`, `ExpSum`, `MixedSum`, `InfiniteChain`, `InfiniteMPO`
— eight, of which `ExpSum`/`MixedSum`/`InfiniteMPO` are unexported yet appear in public
docstrings and error messages.

Export list vs. docs:
* unexported but documented as public: `expterm`, `MixedSum`, `ExpSum`, `InfiniteMPO`,
  `chain_terms`, `translate`, `unitcell_terms`, `total`, `maxspan`, `windowlattice`.
* exported but the docs' first instruction is "never write one": `IrrepOperator`.
* exported yet written `OpSum.`-qualified throughout the docs: `instantiate`, `irrep_mpo`,
  `spin`, `project`, `matrixunit`. `examples/common.jl`'s 6-line `using OpSum: …` is entirely
  redundant against the export list.

## 4. Abelian vs non-abelian

Genuinely divergent (F-moves), but the *spelling* diverges more than the mathematics forces:

| | abelian | non-abelian |
|---|---|---|
| site order in `couple` | free (legs inserted, R-symbol supplied) | must be increasing |
| ≥ 3 operands | variadic `couple(a, b, c, d)` | must nest, naming every `to` |
| `dot` | either order | either order |
| on-site builder | `spin_ops(V, up, dn)` → `(; Sp, Sm, Sz)` | `spin(V)` → one operator |

`examples/multibody.jl` discovers the legal channel tuples by a `try`/`catch` loop over
`j12 ∈ 0:2, j123 ∈ 0:3` — that loop is evidence of a missing query (`couple_channels(ops…; to)`).

## 5. Docs coverage gaps

The README advertises the example series as the tour. It stops at fermions:

1. **no infinite-chain example page** (feature #28) — only a section in `operators.md`;
2. **no exponential-decay example page** (feature #29) — likewise;
3. **no `jordan_mpo_tensors` / MPSKit handoff page** — "how do I hand this to DMRG" is the first
   question a user has, and it exists only as one paragraph of `operators.md`;
4. no page treating **truncation** (`SVDBondAlgorithm`) as a workflow — it is a subsection of
   `long_range.jl`;
5. no page on **charged operators** (`to != unit(I)`) as a building block, though `couple`
   advertises them;
6. the "Reading an error" table omits the infinite/exp errors (`unitcell_terms` double-counting,
   `expterm` decay bounds), which are the likeliest to be hit by a new user of those features.

## 6. Options for the lattice axis

### Option 1 — lattice at the call site, always: add `FiniteChain`

```julia
irrep_mpo(h, FiniteChain(V, 8))
irrep_mpo(h, InfiniteChain([V]))
irrep_mpo(h + expterm(h; decay = 0.4), FiniteChain(V, 8))
irrep_mpo(h + expterm(h; decay = 0.4), InfiniteChain([V]))
```

One rule: terms are latticeless, the lattice is the second argument, there are two lattices.
+ perfect uniformity over the grid; the `Terms`/`TermSum`/`MixedSum` distinction stops leaking
  into the top-level call.
− `TermSum` loses its raison d'être, but `opsum`'s eager letter-vs-space check and `append!`
  accumulation are worth keeping, so both spellings survive — two idioms again.

### Option 2 — lattice in the operator, always: `opsum` becomes universal

```julia
H = opsum(fill(V, 8), h)                                    # finite
H = opsum(InfiniteChain([V]), h)                            # infinite
H = opsum(fill(V, 8), h, expterm(h; decay = 0.4))           # + channels
H = opsum(InfiniteChain([V]), h, expterm(h; decay = 0.4))
irrep_mpo(H)                                                # always one argument
```

This makes CLAUDE.md's stated principle ("there is no useful latticeless-but-compressible
state") actually universal instead of finite-only.
+ `H'`, `islossless`, `length`, `canonicalize!` become available on every path for free — the
  functional gap in §2 closes as a side effect;
+ `irrep_mpo` drops to one positional argument, so `alg` stops competing with the lattice for
  slot 2.
− needs `TermSum{I, Lat}` carrying an (often empty) `ExpSum`; bigger refactor;
− `instantiate` on an infinite `TermSum` has no meaning and must throw, so "everything works on
  a `TermSum`" is not quite true;
− `opsum(::InfiniteChain, …)` must skip the `1:N` site check and (lazily) run `unitcell_terms`.

Options 1 and 2 are the same design from either end; the choice is whether the lattice lives in
the operator or at the call site.

### Option 3 — minimal patch: complete the grid, keep both idioms

Add `irrep_mpo(::Terms, sites)`, `opsum(::InfiniteChain, …)`, `opsum(sites, ::MixedSum)`,
`adjoint(::Terms, sites)`; fix the export list.
+ cheap, non-breaking, every doc example stays valid, and the A→C trap disappears.
− two idioms forever; the four-type zoo stays user-visible.

## 7. Options for the coupling axis

### Option 4 — one variadic `couple`, channels optional exactly when forced

```julia
couple(cd[1], c[2], cd[3], c[4])            # abelian: every channel forced, nothing to say
couple(S[1], S[2], S[3]; channels = (1,))   # non-abelian: name the inner lines
couple(S[1], S[2], S[3])                    # error listing the legal channel tuples
couple_channels(S[1], S[2], S[3]; to = 0)   # and a query that returns them
```
+ one spelling for both symmetry classes; nesting becomes an implementation detail; kills the
  `try`/`catch` discovery loop in `multibody.jl`.
− `channels = (1, 0)` is positional-by-convention, less self-documenting than nested `to =`;
− hides the caterpillar, which the present API deliberately exposes.

### Option 5 — uniform on-site builders

`spin_ops(V)` for every symmetry, returning `(; S)` under SU(2) and `(; Sp, Sm, Sz)` under U(1),
so model code always opens with one call. Retire the bare `spin`, or keep it as an alias.
+ one entry point; the field names then document which symmetry you are in.
− the NamedTuple's fields depend on `sectortype(V)`, so model code still is not symmetry-generic.

Open question behind this: is OpSum meant to ship *model* builders (a `heisenberg_bond(V, i, j)`
that dispatches on `sectortype`), or only the algebra? That is the only route to genuinely
symmetry-agnostic model code, and it is a scope decision, not an API one.

## 8. Recommendation

Option 2 for the lattice (it closes §2's functional gap rather than just tidying), Option 3 as
its non-breaking first step, plus the export-list fix and three new example pages: infinite
chains, exponential decay, and the MPSKit / `jordan_mpo_tensors` handoff. Option 4 is a separate
and smaller decision; Option 5 is cosmetic and can ride along.

## 9. The lattice, revisited: what actually needs it

Two facts checked against the source.

**(i) The compression is space-free.** `ITOTermTable(H)` touches the lattice exactly once, as
`N = length(lattice(H))` (`irreptermtable.jl:102`); `irrepgraph.jl`, `irrepgraph_vc.jl` and
`irrepgraph_svd.jl` never mention an `ElementarySpace` at all. So `irrep_mpo`'s dependence on the
lattice is a dependence on a **site count**, nothing more. `mpo_terms`' docstring already says as
much from the other side — the bond data "names charges but not physical spaces".

**(ii) `SiteOperator{I}` does not store its space.** It is letters + coefficients, sector-typed
only. That is precisely why `adjoint` and `instantiate` need an external lattice, and why
`opsum` has to exist as the place where operators and spaces are confronted.

So the real layering is:

| needs | what |
|---|---|
| nothing but site indices and charges | term algebra, `couple`, `canonicalize!`, `≈` |
| the site **count** `N` (or period `L`) | the whole sweep, `irrep_mpo`, `mpo_terms` |
| the **spaces** | letter validation, `adjoint`, `instantiate`, `irrep_mpo_tensors` |

The lattice enters at `opsum` today, two layers earlier than anything requires. And note that
`MixedSum` — the newest feature — already takes its lattice at `irrep_mpo`. The exception is
`TermSum`, the oldest type. Deleting the exception is the refactor; the rule already exists.

## 10. Options for "no lattice until the MPO is formed"

### Option A — `irrep_mpo` takes the lattice, `TermSum` retires

`Terms` (and `MixedSum`) become the only operator types. `opsum` keeps its name but loses its
first argument: it is the **one-pass accumulator**, which is what it was always doing — the name
means "operator sum", not "operator sum on a lattice".

```julia
h = opsum(                                    # Θ(M), one pass, latticeless
    (J1 * dot(S[i], S[i + 1]) for i in 1:(N - 1)),
    (J2 * dot(S[i], S[i + 2]) for i in 1:(N - 2)),
)
append!(h, more_terms)                        # still linear

irrep_mpo(h, fill(V, N))                      # or FiniteChain(V, N)
irrep_mpo(h, InfiniteChain([V]))
irrep_mpo(h + expterm(h; decay = 0.4), fill(V, N))
irrep_mpo(h + expterm(h; decay = 0.4), InfiniteChain([V]))
islossless(h, fill(V, N))
instantiate(h, fill(V, N))                    # this signature already exists
```

+ one rule, all four cells; `TermSum` and `ExpSum`/`MixedSum`'s asymmetry both go away;
+ the confront-once guarantee is **not** lost — validation moves from `opsum` to `irrep_mpo`,
  still exactly one place, still before any output is produced;
− `H'` sugar dies: `adjoint` genuinely needs spaces, so it becomes `adjoint(h, spaces)`. For the
  finite user that is a regression on today.

### Option B — let the spaces ride on the operators

Every `SiteOperator` is already **born from a space**: `spin(V)`, `matrixunit(V, out, in)`,
`project(O, V)`, `spin_ops(V, …)`, `fermion_ops(V)`, `scalarop(c, V)`. The space is at the call
site already and is then thrown away. Keep it, and a `Term` becomes self-describing for the sites
it touches.

```julia
S = spin(V)                                   # carries V
h = dot(S[1], S[2]) |> opsum                  # the bag knows: site 1 → V, site 2 → V

h'                                            # works, no lattice
instantiate(h)                                # works, no lattice
irrep_mpo(h, N)                               # only the count
irrep_mpo_tensors(Ws, secs, fill(V, N))       # spaces only for the *idle* sites
```

+ this is the only option that gets the lattice out of the operator sum **without** losing `H'`;
+ validation moves *earlier* than today — to placement, the best possible error locality;
+ the "mismatched `sites` vector" failure mode disappears for every active site;
+ `fill(V, N)` boilerplate largely evaporates.
− `SiteOperator` is the element type of the reduced bond matrices, and the sweep builds those
  from `ITOKey`s, which have no space — so a space-carrying `SiteOperator` would have to be
  threaded through the sweep or the bond type would have to differ from the builder type;
− arithmetic between operators on different spaces has to throw; memo keys change.

### Option B″ — the same, but confine the space to the builder output

Keep `SiteOperator{I}` exactly as is (space-free algebra type, bond-matrix element type) and have
the builders return a thin `SpacedOperator{I, S} = (op, V)` that `getindex` consumes:

```julia
S = spin(V)          # ::SpacedOperator{SU2Irrep, …}
S[1]                 # ::Terms, recording site 1 → V
```

+ all of B's wins, and the sweep and bond-matrix types are untouched — B's one real objection
  disappears;
− one more type in the zoo, and two on-site types (`SiteOperator` vs `SpacedOperator`) whose
  difference a user will occasionally have to understand.

### Sub-decision: what is the lattice argument?

Independently of A/B: a bare `Vector{<:ElementarySpace}`, an `Int` (finite, spaces deferred to
assembly), or a `FiniteChain` type parallel to `InfiniteChain`. A `FiniteChain(V, N)` constructor
would also kill the `fill(V, N)` idiom, and makes `irrep_mpo(h, lat)` read the same in both
cells. Relatedly: `irrep_mpo` should probably return a `FiniteMPO` struct rather than a bare
tuple, so the finite and infinite returns are the same kind of thing.

## 11. Revised recommendation

Option A is the right destination for the lattice, and `B″` is what makes it free rather than a
trade: A alone regresses `H'` for finite users, and `B″` turns that regression into an
improvement while also moving validation earlier. Sequence: A first (it is mostly deletion),
`B″` second, `FiniteChain` + struct returns alongside either.

## 12. Decision and what landed

**Decided: Option A**, with `FiniteChain` for the lattice and a `FiniteMPO` struct for the return.
Option B was rejected on efficiency grounds as much as design ones — the spaces have no business
riding along through the MPO compression, which (fact (i) above) never looks at one. Losing the
postfix `H'` was accepted as the price; `adjoint(h, lat)` replaces it.

What changed:

* **`TermSum` is gone.** `Terms` is the compressible operator, and it takes over `TermSum`'s
  canonicalising accessors — `length`, iteration, indexing, `≈`, `show` all normalise first, with
  `nterms_raw` for the append count.
* **`opsum` lost its first argument** and is now purely the one-pass accumulator, which is all it
  ever was; the name means "operator sum", not "operator sum on a lattice". `opsum(sites, …)`
  throws an `ArgumentError` naming the replacement rather than a `MethodError`. Establishing the
  sector type costs one dynamically dispatched step on the first term, after which the typed
  collector runs — so it is still `Θ(M)` with the old constants.
* **`src/operators/lattices.jl`** holds `AbstractLattice`, `FiniteChain` and `InfiniteChain` (moved
  out of `infinite/infinitechain.jl`, which keeps the term-level operations). `FiniteChain(V, N)`
  replaces `fill(V, N)`. The two differ in exactly two observable ways: whether `getindex` wraps,
  and whether a site outside `1:length` is an error.
* **`irrep_mpo(h, lat[, alg])`** is the single entry point for all four cells of the grid, and is
  where `_checklattice` runs — so the confront-once guarantee survives the move from `opsum`
  intact. A lattice may be either chain type or any iterable of one space per site.
* **`FiniteMPO`** mirrors `InfiniteMPO`: same fields, same `Ws, secs = mpo` destructuring, a `show`
  that prints the bond profile. Neither got a `getindex` — `length` is the site count, so indexing
  1..2 to mimic the old tuple would have been a wart; call sites use `.bondsectors`.
* **`mpo_terms(Ws, secs)` lost its `sites` argument** entirely. It only ever used it to build the
  `TermSum`; the bond data names charges and not spaces, so the natural return is a latticeless bag.
* **`ITOTermTable(ts, N)`** takes the count explicitly, which is the one thing the sweep wanted from
  the lattice all along.
* `instantiate(h, lat)`, `islossless(h, lat[, alg])`, `jordan_mpo_tensors(h, lat[, alg])`,
  `adjoint(h, lat)`; exports gained `FiniteChain`, `FiniteMPO`, `InfiniteMPO`, `AbstractLattice`,
  `expterm`, `ExpSum`, `MixedSum` and lost `TermSum`, `lattice`.

Three things the migration itself taught us, each now fixed or recorded:

1. A generic `irrep_mpo(h, lat)` fallback happily swallowed a **misplaced algorithm selector** as a
   lattice and failed inside `collect`. `_tolattice` now guards on iterability first and names what
   was wrong.
2. The **sector-type check** `opsum` used to do (`_wrongsector`) had no home after the move; without
   it an operator over the wrong symmetry failed deep in the alphabet lookup with a `convert` error.
   `_checklattice` does it now.
3. Downstream code that wants to pass one object around has to pair the two itself. The benchmark
   model builders now return `(h, lat)`, and `examples/common.jl:build` takes both. That is the
   honest cost of Option A, and it is small — but it is real, and it is the shape every caller
   who wants a single value will land on.
