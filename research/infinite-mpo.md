# Infinite MPOs with a repeating unit cell — design note

*What survives from the finite pipeline when the chain becomes infinite, what has to be adapted, and
why. Companion to `research/persistent-graph-mpo.md`, whose §2 machinery this builds on directly.*

New files: `src/operators/infinitechain.jl` (the lattice descriptor, `translate`, canonicalisation) and
`src/operators/infinitegraph.jl` (the window construction, identity-channel detection, faithfulness).
`irrepgraph.jl` gains the canonicalisation described in §3; `irrepmpo.jl` gains the public entry and
the wrap-around tensor assembly. The sweep itself — `_at_site!`'s five phases — is **unchanged**.

Semantics, fixed once:

```julia
irrep_mpo(H, InfiniteChain(spaces))   represents   Σ_{n ∈ ℤ} translate(H, n·L),   L = length(spaces)
```

`H` is a *generating set*, one representative per translation class, not the Hamiltonian.

## 1. What survives

| Layer | Verdict |
|---|---|
| `Term` / `Terms` / `TermSum` / `couple` / `dot` / `scale` | **verbatim.** A `Term` stores plain `Int` sites, which already range over all of ℤ. The generating set stays a latticeless `Terms` bag — the `InfiniteChain` names the space of every site by wraparound, so there is nothing for `opsum` to bind. |
| `project` / `instantiate` / `matrixunit` | **verbatim.** Purely local. |
| `ITOKey`, `passthrough`, caterpillar helpers | **verbatim.** |
| `min_vertex_cover_bipartite` (Hopcroft–Karp + König) | **verbatim.** |
| `bipartite_connected_components` | **verbatim.** |
| `_at_site!`, `_vc_component`, the covered-left/covered-right coefficient flow | **verbatim** (`_vc_component` gained a per-index `origins` return, used only for ordering). |
| Per-bond-sector block-diagonality (`ITOKey.bond` purity) | **verbatim**, given the neutrality scope limit (§2). |
| `_promote_pending!` / the sentinel | **verbatim, and promoted from corner case to load-bearing** — §4. |
| `irrep_mpo_tensors` | **one line**: `bsecs[1] = bondsectors[L]` instead of `I[unit(I)]`. Factored into `_mpo_tensors`. |
| `mpo_terms` | **two keywords** (`leftidx`, `rightidx`) for the boundary vectors. |
| `_suffix_ids` interning | **needs a twin.** It keys on the *absolute* site, so it can never see two bonds one cell apart as posing the same problem. §3. |
| Adjacency order (`_merge_edges!` first-encounter) | **needs canonicalising.** Fine for one pass, fatal for a fixed point. §3. |
| Boundary seeding (`nlinks = 1`, `rbond = unit(I)`) | **replaced** by a fixed point with both identity channels live at every bond. §5. |
| `_irrep_svd` / `SVDBondAlgorithm` | **does not extend.** Compresses each bond independently against a hard-coded `N-1`-internal-bond, vacuum-terminated layout (`irreptermtable.jl:361-394`); there is no bond basis that closes on itself. Throws. |

The reframe that organises all of it: **the suffix-class structure is a DAG today** — bottom-up
hash-consing over a finite chain, `sufid[j,t] = intern((site, key, sufid[j+1,t]))`. Finite range on a
periodic lattice keeps it acyclic (bounded relative offsets ⇒ a finite, `N`-independent name set) and
needs only a change of *naming*. §7's exponentially decaying terms are what make it genuinely cyclic —
a finite automaton rather than a DAG — and that is the one place a change of *kind* is required.

## 2. Scope

* **`total(term) == unit(I)` for every term.** `rbond` is an absolute running fusion outcome measured
  from the left vacuum. Under wrap-around the translation-invariant notion is "charge accumulated since
  the class entered", and only for a neutral Hamiltonian do the two coincide (the reference is
  `unit(I)` everywhere). Charged infinite MPOs are out of scope and rejected.
* **`K = 0` identity terms rejected**: `Σ_n c·𝟙` does not converge.
* Existing limits carry over: multiplicity-free fusion, no on-site `Prod`/`Pow`, no out-of-order
  `couple`, no fermionic/graded sectors on the graph path.

## 3. The `L = 1` hazard, and the canonicalisation it forces

This is the part the note exists for, and it is the first thing that actually broke.

Hopcroft–Karp's matching — and therefore König's cover — is a deterministic function of the order the
adjacency lists are scanned in. That order was deliberately history-dependent: `_merge_edges!` keeps
first-encounter order, with the comment that "nothing downstream needs it sorted"
(`irrepgraph.jl:287-288`). On a finite chain that is exactly right: every minimum vertex cover is as
good as any other, and the tests compare per-sector multiplicities and `mpo_terms` round-trips rather
than raw matrices precisely because the choice is not canonical.

On a periodic lattice it is fatal. At `L = 1` every bulk bond poses the *same* problem, so the sweep
must give the *same* answer, and "the same" has to mean identically labelled, not merely isomorphic —
otherwise the extracted cell's left and right bond bases are ordered differently and it does not tile.
Measured on the very first model tried (`L = 1` SU(2) Heisenberg, unrolled to 88 cells), consecutive
bulk bonds agreed on charges `[0, 1, 0]` at every single bond and never once produced identical
reduced tensors: the cover flip-flopped between covering the exhausted class on the right and covering
the identity-backbone left vertex, which moves the term's coefficient between the two.

Two halves fix it, both additive and both always-on:

**(a) Translation-invariant class names** (`_rel_suffix_ids`, `_rdesc`). `_suffix_ids` conses
`(absolute site, key, tail)`; its twin conses `(gap to the next active site, key, tail)` — shape with
position factored out. A right vertex's canonical name at bond `i` is then the triple

```
(distance from i to the first remaining factor, shape id, running bond charge)
```

which is equal for a class and its `L`-translate at bond `i + L`. The absolute form is kept: within one
bond the two agree, it is cheaper, and `pendbysig` keys on it.

**(b) Canonical order** (`_canonicalise_rights!`, `_canonicalise_bond!`). Right vertices are renumbered
ascending in that name and every adjacency list is sorted; the assembled bond is ordered covered-left
first (by `(link, key)`) then covered-right (by class name). The induction that makes the sort keys
themselves translation-invariant is that after the pass *an index's position is its canonical rank*, so
`link` needs no further translation — seeded by the one-dimensional boundary bond.

Neither changes any bond dimension. What they remove is the history dependence, and the payoff is
visible in two ways beyond the infinite construction: bulk bonds of an *unrolled* sweep become equal
entry for entry, and the reduced MPO of a given Hamiltonian no longer depends on the order its terms
were written in (both pinned in `test/test_infinite_graph.jl`). The whole finite suite (2959 tests)
passes unchanged.

**Still open.** The per-bond cover is chosen greedily, given the previous bond. For a cyclic automaton
locally minimal per bond need not be globally minimal, and canonicalisation does not address that — it
makes the choice *consistent*, not *optimal*. No model in the suite shows a gap, but none of this is a
proof.

## 4. The collision was the wrap-around identification all along

`research/persistent-graph-mpo.md` §2.2 flags `_promote_pending!` as "the one invariant a change here is
most likely to break": `_op_at_ito` fills idle sites with a pass-through carrying the *running* charge,
so a started term whose charge has fused back to `unit(I)` is indistinguishable over its idle sites from
one that has not started, and when its remaining factors coincide with a pending term's whole content
the two classes are genuinely equal. The note observes that **no showcase model triggers it**.

On a periodic lattice it is the common case. There is always a translate further right, so
`nremaining > 0` forever, the sentinel is present at every bond, and the start channel never
disappears; and any on-site field alongside a two-site interaction makes the probe fire at *every*
bond. The mechanism the finite pipeline needed a hand-written counterexample to exercise is the one
doing the routine work here. It needed no change at all — only relative naming (§3a), which is what
lets the probe match the *nearest* translate to the right rather than a particular absolute one.

It also has an observable consequence worth stating, because it makes the obvious faithfulness test
wrong. When a term's suffix class is shared with a longer term, its coefficient is folded onto the
shared channel's **trailing pass-through** rather than onto its own last site. For U(1) XXZ plus an
`Sᶻ` field the cell is

```
(start, chan) = Sᶻ letter          # the field's letter goes down at site s
(chan, done)  = 0.15·𝟙 + …         # its coefficient is picked up on the identity at site s+1
```

so a *path* can run up to `R` sites past the support of the term it represents. §6 says what that does
to the test.

## 5. Closing the bond: the window construction

The sweep is single-pass in four places, three of which are boundary folding (`i == N` in
`_vc_component` twice, and the `i < g.N` guard on `_build_next_graph!`) and one of which is structural
(the monotone cursor is monotone in *absolute* site). Rather than rewrite the sweep to run cyclically,
the construction unrolls a window and reads the middle cell off:

1. generate every `L`-translate whose support fits inside `ncells` cells (`window_terms`);
2. run the **unchanged** sweep, `_irrep_sweep(tt, N, VertexCover())`;
3. find the first cell that has reached the fixed point and return it.

Step 3 is a *search*, not an offset formula, and that matters. The bond bases converge within the
interaction range `R`, but the residual coefficients riding on them take longer: the identity backbone
on the right does not exist until something has finished, so the first completed term's coefficient sits
on the done channel until a covered-right reset normalises it. On `L = 1` Heisenberg the bases are final
at bond 2 and the tensors only at bond 4. `_fixedpoint_cell` therefore requires a cell and the **two**
after it to be identical entry for entry with matching bond charges, skips cells within `R` of the left
edge (where the window has dropped the translates that stick out), and the window doubles twice before
giving up.

Cost is `Θ(M_gen · ncells)` term generation plus a linear sweep over `ncells · L` sites, with
`ncells = Θ(R/L)` — negligible for a finite-range model, and it buys reuse of the entire tested sweep
instead of a second implementation of it.

**Identity channels.** An infinite MPO is unusable without its two boundary vectors, so the
construction reports them. They are found on the assembled cell rather than inside the cover, by
*direction*: both are chains of bare pass-through entries running once around the cell, and nothing
*enters* the start channel while nothing *leaves* the done channel. Detection throws if either is
missing, ambiguous, or not charge-neutral. Doing it on the output rather than tracking the exhausted
class through `_vc_component` avoids a genuine edge case: König can leave the exhausted class uncovered
when several left vertices feed it, in which case the identity backbone spreads over more than one bond
index and there is no single done channel. No model in the suite does this; if one ever does, detection
reports the candidates instead of silently picking one.

Comparing `Ws` needs care. `SiteOperator` does have a structural `==`, but it compares its two parallel
`letters`/`coeffs` vectors *in order*, and `+` accumulates in insertion order, so two separately built
copies of the same entry can disagree on letter order — and the fixed-point check compares entries
built at different points in the sweep. `_canonform` sorts each entry's `letter => coeff` pairs by
letter before comparing. (Against the older `LocalOp`, which had no structural `==` at all, the first
version of this check reported "never converges" for the stronger version of the same reason.)

## 6. Verification

* **`mpo_terms_window`** tiles the cell, walks from the start channel and accepts paths landing on the
  done channel, and is compared against `window_terms`. It is a **sandwich**, not an equality, for the
  reason in §4: everything produced must be a translate at its exact coefficient (soundness), and
  everything whose support ends `R + 1` sites before the right edge must be produced (completeness away
  from the edge). This pins coefficients *and* caterpillar fusion trees, not just dimensions.
* **`contract_open`** caps both boundary bonds with one-hot maps onto the two channels and contracts
  the tiled tensors down to an operator, compared against `instantiate` of the terms the MPO claims to
  generate — the infinite counterpart of `examples/common.jl`'s `mpo_tensormap`, which can only
  `removeunit` a one-dimensional vacuum.
* **Unit-cell invariance** is the sharp translation-covariance test: the same model written on cells of
  1, 2, 3 and 4 sites must give the *same* MPO site for site. `L = 1` is the hardest case, `L > 1` with
  range `> L` the next hardest.
* **Order independence**: building the same Hamiltonian from a shuffled term list gives a different
  `TermSum` insertion order, hence different right-vertex ids — and now an identical MPO.
* Reference numbers: `L = 1` SU(2) Heisenberg is the textbook `[0, 1, 0]`, dense `D = 5`; adding a
  `k`-th neighbour coupling costs one spin-1 channel each, `D = 3k + 2`.

## 7. Exponentially decaying terms — design, not yet built

`Σ_{i<j} λ^{j-i-1} A_i B_j` (optionally with a string operator on the intermediate sites) is the second
regime. Its Jordan-MPO signature is a scalar `λ` on the **diagonal** of a bond channel, and it cannot be
enumerated as a flat term list at all.

**Representation.** A primitive alongside the generating `TermSum` — it cannot live in a `K×M` table —
of the form `expterm(A, B; decay = λ, string = 𝟙)`. Require `|λ| < 1`; `λ = 1` is not summable and must
error.

**A geometric channel is a suffix class with a self-loop.** That is the whole design: make it an
ordinary right vertex whose name is self-referential (self-edge `λ·passthrough → itself`, exit edge
`B → exhausted`), and the existing machinery does the rest —

| case | outcome | mechanism |
|---|---|---|
| same `λ`, same exit, different entry | merge | equal suffix class (`_suffix_merge!`) |
| same `λ`, same entry, different exit | merge | shared left vertex, covered by the min vertex cover |
| different `λ` | stay distinct | correct: the channels are linearly independent |

It also needs no forcing into the bond basis. The channel regenerates itself as the covered-left vertex
`(link = g, key = λ·passthrough)`, which *is* the fixed point — the same structure that makes the start
and done channels persist.

**The one change of kind.** Bottom-up hash-consing cannot name a cyclic tail: `_suffix_ids` and
`_rel_suffix_ids` both terminate because they cons onto an already-interned tail, and a self-reference
has none. The replacement is **partition refinement** (Hopcroft–Moore), the coinductive dual: start
with all classes equal and split by `(next-site key, successor class)` until stable, identifying
classes up to bisimulation. `_suffix_merge!` is already one refinement step per bond — it just refines
*by a precomputed name* rather than *toward a fixed point*.

**Gotcha to remember.** The diagonal must carry the pass-through *letter* weighted by λ, not be an
empty operator: tensor assembly iterates `pairs(localop)`, so a `SiteOperator` with no letters
contributes nothing and would be silently dropped. Under `SiteOperator` this is much harder to get
wrong than it was under `LocalOp` — the bare identity is the `passthrough` sentinel letter rather than
a letter-less scalar variant, so `scalarop(λ, I)` already *is* `λ · passthrough`.

**Out of scope even then**: Jordan blocks (polynomial × exponential decay), sum-of-exponentials fits
for power laws — which is the only route to `1/r^α` on an infinite lattice, since the exact treatment
that gives linear bond growth on a finite chain (`examples/long_range.jl`) has no thermodynamic limit —
and `SVDBondAlgorithm` on channels with a diagonal.

## 8. Follow-ups

* A native cyclic sweep (no window) was planned as the default, with the window as its parity oracle.
  **Decided against.** The window search *is* the fixed-point iteration, so the second implementation
  would only save `Θ(R/L)` cells of unrolling; and it would share `_at_site!`, `_vc_component`, the
  cover and the canonicalisation — i.e. everything that could be wrong — so as an oracle it is much
  weaker than the `mpo_terms_window` round-trip and the `contract_open` dense check, which is what §6
  ended up resting on. Revisit only if the unrolling cost ever shows up in a profile.
* **Performance of the canonicalisation is unmeasured.** §3 adds an `O(live)` name build plus a
  possible sort per bond (`_canonicalise_rights!`) and an `O(D log D)` sort plus a block-dictionary
  rebuild per bond (`_canonicalise_bond!`) to the *shared* default path. Asymptotically that is a log
  factor on an already-dominated term for a finite-range model and `Θ(N² log N)` against an intrinsic
  `Θ(N³)` for an all-to-all one, so it should not move the exponents — but no benchmark was run, so the
  concrete timings and fitted exponents in `persistent-graph-mpo.md` §2.3, and the checked-in figures
  under `docs/src/assets/`, date from before it. Re-run `benchmark/run.jl --sweep ci` before trusting
  them.
* Global vs per-bond cover minimality on a cycle (§3).
* The spread-identity-backbone edge case in `_identity_channels` (§5).
* A canonically *ordered* `SiteOperator`, which would let `_canonform` go away (§5). The structural
  `==`/`hash` this line used to ask for now exists; only the letter ordering is still insertion-order.
* Everything in §7.
* Inherited from `persistent-graph-mpo.md` §5: fermionic/JW strings, `GenericFusion` multi-channel.
