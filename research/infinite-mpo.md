# Infinite MPOs with a repeating unit cell — design note

*What survives from the finite pipeline when the chain becomes infinite, what has to be adapted, and why.
Companion to `research/persistent-graph-mpo.md`, whose §2 machinery this builds on directly.*

New files: `src/operators/infinite/infinitechain.jl` (the lattice descriptor, `translate`, canonicalisation), `src/operators/infinite/infinitegraph.jl` (the window construction, identity-channel detection, faithfulness) and — for §7 — `src/operators/infinite/expterms.jl` (the exponential-decay primitive and its lowering).
`irrepgraph.jl` gains the canonicalisation described in §3 and the geometric right vertices of §7; `irrepmpo.jl` gains the public entries and the wrap-around tensor assembly.
The sweep itself — `_at_site!`'s five phases — is **unchanged** for finite-range models.

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

The reframe that organises all of it: **the suffix-class structure is a DAG today** — bottom-up hash-consing over a finite chain, `sufid[j,t] = intern((site, key, sufid[j+1,t]))`.
Finite range on a periodic lattice keeps it acyclic (bounded relative offsets ⇒ a finite, `N`-independent name set) and needs only a change of *naming*.
§7's exponentially decaying terms are what make it genuinely cyclic — a finite automaton rather than a DAG.
That turned out to need less than a change of kind: the cycle is always *declared*, so one reserved interned id per loop descriptor cuts it and the consing continues above (§7).
What it does need is that the cyclic classes be forced into the bond basis.

## 2. Scope

* **`total(term) == unit(I)` for every term.**
  `rbond` is an absolute running fusion outcome measured from the left vacuum.
  Under wrap-around the translation-invariant notion is "charge accumulated since the class entered", and only for a neutral Hamiltonian do the two coincide (the reference is `unit(I)` everywhere).
  Charged infinite MPOs are out of scope and rejected.
* **`K = 0` identity terms rejected**: `Σ_n c·𝟙` does not converge.
* Existing limits carry over: multiplicity-free fusion, no on-site `Prod`/`Pow`, no out-of-order `couple`, no fermionic/graded sectors on the graph path.

## 3. The `L = 1` hazard, and the canonicalisation it forces

This is the part the note exists for, and it is the first thing that actually broke.

Hopcroft–Karp's matching — and therefore König's cover — is a deterministic function of the order the adjacency lists are scanned in.
That order was deliberately history-dependent: `_merge_edges!` keeps first-encounter order, with the comment that "nothing downstream needs it sorted" (`irrepgraph.jl:287-288`).
On a finite chain that is exactly right: every minimum vertex cover is as good as any other, and the tests compare per-sector multiplicities and `mpo_terms` round-trips rather than raw matrices precisely because the choice is not canonical.

On a periodic lattice it is fatal.
At `L = 1` every bulk bond poses the *same* problem, so the sweep must give the *same* answer, and "the same" has to mean identically labelled, not merely isomorphic — otherwise the extracted cell's left and right bond bases are ordered differently and it does not tile.
Measured on the very first model tried (`L = 1` SU(2) Heisenberg, unrolled to 88 cells), consecutive bulk bonds agreed on charges `[0, 1, 0]` at every single bond and never once produced identical reduced tensors: the cover flip-flopped between covering the exhausted class on the right and covering the identity-backbone left vertex, which moves the term's coefficient between the two.

Two halves fix it, both additive and both always-on:

**(a) Translation-invariant class names** (`_rel_suffix_ids`, `_rdesc`).
`_suffix_ids` conses `(absolute site, key, tail)`; its twin conses `(gap to the next active site, key, tail)` — shape with position factored out.
A right vertex's canonical name at bond `i` is then the triple

```
(distance from i to the first remaining factor, shape id, running bond charge)
```

which is equal for a class and its `L`-translate at bond `i + L`.
The absolute form is kept: within one bond the two agree, it is cheaper, and `pendbysig` keys on it.

**(b) Canonical order** (`_canonicalise_rights!`, `_canonicalise_bond!`).
Right vertices are renumbered ascending in that name and every adjacency list is sorted; the assembled bond is ordered covered-left first (by `(link, key)`) then covered-right (by class name).
The induction that makes the sort keys themselves translation-invariant is that after the pass *an index's position is its canonical rank*, so `link` needs no further translation — seeded by the one-dimensional boundary bond.

Neither changes any bond dimension.
What they remove is the history dependence, and the payoff is visible in two ways beyond the infinite construction: bulk bonds of an *unrolled* sweep become equal entry for entry, and the reduced MPO of a given Hamiltonian no longer depends on the order its terms were written in (both pinned in `test/test_infinite_graph.jl`).
The whole finite suite (2959 tests) passes unchanged.

**Still open.**
The per-bond cover is chosen greedily, given the previous bond.
For a cyclic automaton locally minimal per bond need not be globally minimal, and canonicalisation does not address that — it makes the choice *consistent*, not *optimal*.
No model in the suite shows a gap, but none of this is a proof.

## 4. The collision was the wrap-around identification all along

`research/persistent-graph-mpo.md` §2.2 flags `_promote_pending!` as "the one invariant a change here is most likely to break": `_op_at_ito` fills idle sites with a pass-through carrying the *running* charge, so a started term whose charge has fused back to `unit(I)` is indistinguishable over its idle sites from one that has not started, and when its remaining factors coincide with a pending term's whole content the two classes are genuinely equal.
The note observes that **no showcase model triggers it**.

On a periodic lattice it is the common case.
There is always a translate further right, so `nremaining > 0` forever, the sentinel is present at every bond, and the start channel never disappears; and any on-site field alongside a two-site interaction makes the probe fire at *every* bond.
The mechanism the finite pipeline needed a hand-written counterexample to exercise is the one doing the routine work here.
It needed no change at all — only relative naming (§3a), which is what lets the probe match the *nearest* translate to the right rather than a particular absolute one.

It also has an observable consequence worth stating, because it makes the obvious faithfulness test wrong.
When a term's suffix class is shared with a longer term, its coefficient is folded onto the shared channel's **trailing pass-through** rather than onto its own last site.
For U(1) XXZ plus an `Sᶻ` field the cell is

```
(start, chan) = Sᶻ letter          # the field's letter goes down at site s
(chan, done)  = 0.15·𝟙 + …         # its coefficient is picked up on the identity at site s+1
```

so a *path* can run up to `R` sites past the support of the term it represents.
§6 says what that does to the test.

## 5. Closing the bond: the window construction

The sweep is single-pass in four places, three of which are boundary folding (`i == N` in `_vc_component` twice, and the `i < g.N` guard on `_build_next_graph!`) and one of which is structural (the monotone cursor is monotone in *absolute* site).
Rather than rewrite the sweep to run cyclically, the construction unrolls a window and reads the middle cell off:

1. generate every `L`-translate whose support fits inside `ncells` cells (`window_terms`);
2. run the **unchanged** sweep, `_irrep_sweep(tt, N, VertexCover())`;
3. find the first cell that has reached the fixed point and return it.

Step 3 is a *search*, not an offset formula, and that matters.
The bond bases converge within the interaction range `R`, but the residual coefficients riding on them take longer: the identity backbone on the right does not exist until something has finished, so the first completed term's coefficient sits on the done channel until a covered-right reset normalises it.
On `L = 1` Heisenberg the bases are final at bond 2 and the tensors only at bond 4.
`_fixedpoint_cell` therefore requires a cell and the **two** after it to be identical entry for entry with matching bond charges, skips cells within `R` of the left edge (where the window has dropped the translates that stick out), and the window doubles twice before giving up.

Cost is `Θ(M_gen · ncells)` term generation plus a linear sweep over `ncells · L` sites, with `ncells = Θ(R/L)` — negligible for a finite-range model, and it buys reuse of the entire tested sweep instead of a second implementation of it.

**Identity channels.**
An infinite MPO is unusable without its two boundary vectors, so the construction reports them.
They are found on the assembled cell rather than inside the cover, by *direction*: both are chains of bare pass-through entries running once around the cell, and nothing *enters* the start channel while nothing *leaves* the done channel.
Detection throws if either is missing, ambiguous, or not charge-neutral.
Doing it on the output rather than tracking the exhausted class through `_vc_component` avoids a genuine edge case: König can leave the exhausted class uncovered when several left vertices feed it, in which case the identity backbone spreads over more than one bond index and there is no single done channel.
No model in the suite does this; if one ever does, detection reports the candidates instead of silently picking one.

Comparing `Ws` needs care.
`SiteOperator` does have a structural `==`, but it compares its two parallel `letters`/`coeffs` vectors *in order*, and `+` accumulates in insertion order, so two separately built copies of the same entry can disagree on letter order — and the fixed-point check compares entries built at different points in the sweep.
`_canonform` sorts each entry's `letter => coeff` pairs by letter before comparing.

## 6. Verification

* **`mpo_terms_window`** tiles the cell, walks from the start channel and accepts paths landing on the done channel, and is compared against `window_terms`.
  It is a **sandwich**, not an equality, for the reason in §4: everything produced must be a translate at its exact coefficient (soundness), and everything whose support ends `R + 1` sites before the right edge must be produced (completeness away from the edge).
  This pins coefficients *and* caterpillar fusion trees, not just dimensions.
* **`contract_open`** caps both boundary bonds with one-hot maps onto the two channels and contracts the tiled tensors down to an operator, compared against `instantiate` of the terms the MPO claims to generate — the infinite counterpart of `examples/common.jl`'s `mpo_tensormap`, which can only `removeunit` a one-dimensional vacuum.
* **Unit-cell invariance** is the sharp translation-covariance test: the same model written on cells of 1, 2, 3 and 4 sites must give the *same* MPO site for site.
  `L = 1` is the hardest case, `L > 1` with range `> L` the next hardest.
* **Order independence**: building the same Hamiltonian from a shuffled term list gives a different `TermSum` insertion order, hence different right-vertex ids — and now an identical MPO.
* Reference numbers: `L = 1` SU(2) Heisenberg is the textbook `[0, 1, 0]`, dense `D = 5`; adding a `k`-th neighbour coupling costs one spin-1 channel each, `D = 3k + 2`.

## 7. Exponentially decaying terms — built

`Σ_{i<j} λ^{j-i-1} A_i B_j` (optionally with a string operator on the intermediate sites) is the second regime.
Its Jordan-MPO signature is a scalar `λ` on the **diagonal** of a bond channel, and it cannot be enumerated as a flat term list at all.
New file: `src/operators/infinite/expterms.jl` (the primitive, its containers, the explicit expansion, and the lowering); `irrepgraph.jl` gains geometric right vertices; `infinitegraph.jl` and `irrepmpo.jl` gain the entry points.
Tests: `test/test_exp_decay.jl`.

The central claim: **a geometric channel is a suffix class with a self-loop**, so it is an ordinary right vertex and the existing merge / cover / canonicalisation machinery compresses it.

**Representation.**
`expterm(t::TermSum; decay = λ, exitsite, string = nothing)` takes **one fully specified representative term** and stretches the gap just before `exitsite` geometrically:

```julia
expterm(dot(S[1], S[2]); decay = 0.5)                                    # Σ_{i<j} λ^{j-i-1} S_i·S_j
expterm(couple(Sp[1], Sm[2]); decay = 0.5, string = 2Sz)                 # with a string operator
expterm(couple(couple(S[1],S[2];to=1), S[3]); decay = 0.5, exitsite = 3) # two-site entry block
```

Because the representative is a `TermKey` it already carries the caterpillar tree, so *every* fusion channel is named and no charge bookkeeping had to be invented — that is what makes multi-site entry and exit blocks nearly free.
`ExpSum` collects channels, `TermSum + ExpSum` gives a `MixedSum`, and `irrep_mpo` takes that on an `InfiniteChain` *or* on a finite `sites` vector.
The lattice supplies the translation period `P` (`L`, or 1 on a finite chain, where the model is the geometric sum truncated to the chain — spelled out by `chain_terms`).
`λ` counts **per site**; `0 < |λ| < 1` is enforced, and the string is required to be charge-neutral, or the running bond charge would drift along it and the loop would not close on itself.

**The naming: no partition refinement.**
Bottom-up hash-consing can name a cyclic tail because cyclic tails only ever come from a *declared* primitive: reserving **one interned id per loop descriptor** — `(λ, string transitions, exit key, exit-class name, period, δ)`, which fixes the entire cyclic future in closed form — cuts the cycle, and ordinary consing works again on top of it.
Name equality ⟺ class equality still holds (a geometric class can never be bisimilar to a finite one: infinite versus finite support), so `_suffix_merge!` and the canonical order need nothing new.
Partition refinement only becomes necessary if cyclic tails ever become *compositional* — a channel exiting into another channel, i.e. Jordan blocks.

Each channel is lowered to a small weighted automaton (`_lower_channels`) whose states are exactly the suffix classes it can occupy: `E_b` (entry block partly placed), `W_δ` (mandatory waits when the representative's gap exceeds the period), `D_δ` for `δ = L … 1` (**the cyclic core**, `δ` = distance to the next legal exit) and `X_b` (exit block partly placed, landing on the exhausted class).
Keeping `δ` in the state is what makes the phase alignment of an `L > 1` cell fall out: the sweep does no phase arithmetic at all.
A right vertex is then a `(channel, state)` pair, `_signature!` returns the state's name in a disjoint negative id space, and `_rdesc` gained a leading `kind` field.
The synthetic exhausted class deliberately keeps the ordinary `(0, charge)` signature so that it merges with exhausted term classes — that merge *is* the done channel.

**Everything the design table predicted happens** (pinned as bond dimensions in `test_exp_decay.jl`):

| case | outcome | mechanism |
|---|---|---|
| same `λ`, same exit, different entry | merge | equal suffix class (`_suffix_merge!`) |
| same `λ`, same entry, different exit | two classes, one entry column | shared left vertex, covered by the min vertex cover |
| different `λ` | stay distinct | correct: the channels are linearly independent |

and `L = 1` exponentially decaying Heisenberg costs `[0, 1, 0]`, dense `D = 5` — the channel *replaces* the in-flight spin-1 of the nearest-neighbour model rather than adding to it.

**Forcing the cyclic states into the cover.**
A cyclic state must be forced into the vertex cover: if left uncovered, its predecessor becomes covered-left, which *forwards* the self-edge weight `λ·w` instead of resetting it to 1; the λ powers then ride along the bond instead of landing on the diagonal, and a bond that keeps doing that never repeats.
König really can pick that cover — two equal-size minimum covers exist as soon as a channel shares its entry letter with a finite-range term.
Measured on that model (`dot(S[1],S[2]) + expterm(dot(S[1],S[2]); decay=λ)`) the bad choice *is* made, at the first bond where both classes are live, and then heals one bond later: a covered-left predecessor is itself a second predecessor of the cyclic state, which then has two pendants and must be covered.
So both variants converge on every model in the suite, with identical cells and identical bond dimensions.
`_forced_cover` makes it structural anyway — `{v} ∪ MVC(G∖v)` is minimum among the covers containing `v`, and in the bulk a cyclic state always has a pendant predecessor, so forcing is free there and can cost one index only in the window's discarded boundary cells.

**The λ·pass-through trap was real but is structurally avoided.**
λ is an *edge weight*, never an on-site scalar, so `_vc_component`'s existing `lv.key.op * w` emits the pass-through letter scaled by λ — a one-letter `SiteOperator`, which tensor assembly keeps.
`test_exp_decay.jl` pins the diagonal entry's letter and coefficient directly, because a letter-less entry would have produced a correct-looking reduced MPO and a wrong tensor.
(`SiteOperator` narrows the trap further than `LocalOp` did: the bare identity is the `passthrough` sentinel letter rather than a letter-less scalar variant, so the only way to lose the entry now is to emit an *empty* operator.)
As a bonus, `|λ| < 1` also keeps `_identity_channels` honest: a scaled pass-through is not a *bare* one, so a channel can never be mistaken for an identity backbone.

**Two further considerations.**

* *Pruning.*
  A channel state that can no longer complete inside `1:N` contributes no term, so its edge is dropped when the next graph is built.
  Without it the last bonds of a finite chain would carry live channel states and the right boundary would not be one-dimensional — this is what makes the finite entry point work at all, and it also cleans up the window's right edge.
* *Unit-cell invariance does not extend to channels.*
  A channel's period is part of its declaration, so the same physical all-pairs interaction written on an `L`-site cell needs `L²` channels; their classes collapse to the `L` states `δ = 1 … L`, giving `D = 3L + 2` against the `L = 1` model's 5.
  The terms are identical (the round-trip test checks every cell size); the bond dimension is not.
  Measured, the `L` cyclic columns of the bond coefficient matrix are exactly rank 1 with complete-bipartite support, so recombining them needs a *rank*-aware bond choice and the vertex cover is only support-aware — the same blindness that already makes finite `SᶻSᶻ` cost two channels rather than one.
  Until then, write the smallest cell you can.
  §8 records the two ways out: an exact symbolic normalisation that covers this case only, and rank-aware bond selection, which covers both.

**Verification** (`test/test_exp_decay.jl`, 330 tests): the `mpo_terms_window` sandwich against the explicit expansion (`window_terms` now expands channels) over twelve models; an **explicit truncation** oracle — the same interaction written out to `R_max` — with the geometric bond dimension checked `R_max`-independent while the truncated one grows as `3R_max + 2`; `contract_open` against `instantiate` for four models including a string operator and an `L = 2` cell; the merge table; window-size, term-order and unit-cell independence; the finite chain against `chain_terms`; and the validation errors.

**Still out of scope**: Jordan blocks (polynomial × exponential decay) — the one case that would genuinely need partition refinement; sum-of-exponentials *fitting* for power laws, which is the only route to `1/r^α` on an infinite lattice (the exact treatment that gives linear bond growth on a finite chain, `examples/long_range.jl`, has no thermodynamic limit) — the primitive is built so that a fit is a plain sum of `expterm`s, each costing one channel; and `SVDBondAlgorithm` on channels with a diagonal, which throws.

## 8. Follow-ups

* A native cyclic sweep (no window) was planned as the default, with the window as its parity oracle.
  **Decided against.**
  The window search *is* the fixed-point iteration, so the second implementation would only save `Θ(R/L)` cells of unrolling; and it would share `_at_site!`, `_vc_component`, the cover and the canonicalisation — i.e. everything that could be wrong — so as an oracle it is much weaker than the `mpo_terms_window` round-trip and the `contract_open` dense check, which is what §6 ended up resting on.
  Revisit only if the unrolling cost ever shows up in a profile.
* **Performance of the canonicalisation — measured, and it costs nothing detectable.**
  §3 adds an `O(live)` name build plus a possible sort per bond (`_canonicalise_rights!`) and an `O(D log D)` sort plus a block-dictionary rebuild per bond (`_canonicalise_bond!`) to the *shared* default path.
  Asymptotically that is a log factor on an already-dominated term for a finite-range model and `Θ(N² log N)` against an intrinsic `Θ(N³)` for an all-to-all one, so it should not move the exponents.
  The `--sweep full` run of 2026-09-15 (same host as the previous figures, so the code change is isolated) confirms it: finite-range compression fits `0.92 … 1.08`, long-range `2.17 … 2.20` — both within run-to-run noise — and `docs/src/assets/profile.png` regenerated bit-identical, so no bond dimension anywhere changed.
  `persistent-graph-mpo.md` §2.3 and the checked-in figures are current again.
* Global vs per-bond cover minimality on a cycle (§3).
* The spread-identity-backbone edge case in `_identity_channels` (§5).
* A canonically *ordered* `SiteOperator`, which would let `_canonform` go away (§5).
  The structural `==`/`hash` this line used to ask for now exists; only the letter ordering is still insertion-order.
* From §7, in rough order of value.
  The first two both attack the same surplus — the `L` cyclic states of an all-pairs interaction on an `L`-site cell (`D = 3L + 2` against an achievable 5) — from opposite ends, and are worth keeping distinct: the first is exact, cheap and narrow, the second general but with a contract change attached.
  * **Collapse same-descriptor channels to period 1** — a normalisation pass in `_lower_channels`, no linear algebra involved.
    The `L` states are proportional *by construction*, not by numerical accident: they are the same declared loop at different phases.
    So when the `L²` declarations of a cell cover every phase pair with equal `(λ, string, entry/exit letters, coefficient)` *and* the cell's physical spaces are all equal (a letter `(c, n)` only means the same operator at another phase if the space repeats), replace them by a single channel of period 1 and gap 1.
    `ExpChannel` already carries its own `period`, so channels of mixed period coexist with no change to the sweep, and the merged class is literally the `L = 1` channel.
    Exact, deterministic, no gauge question.
    Does nothing for staggered coefficients, phase-dependent letters, or `SᶻSᶻ`.
  * **Rank-aware bond selection.**
    The general statement of the same gap, and the only route to the `SᶻSᶻ` case, which is a rank question about *letters* rather than about phases.
    Measured, the cyclic block is exactly rank 1 with complete-bipartite support (two identical columns, exact zeros elsewhere), so a column-pivoted QR reveals it on the first pivot — SVD's optimality is not needed, and `_irrep_graph_svd` is already the charge-graded sequential QR-style sweep.
    Four things stand between that and using it here:
    * it must be a column **merge**, not a column **selection**: identical columns mean the operator contains `prefix ⊗ (suffix₁ + suffix₂)`, so an interpolative decomposition that keeps one column and drops the other is wrong.
      Consequence: a bond index stops being *a* suffix class.
      (The geometric structure itself survives fine — the cyclic block is `λ` times the identity, and a basis change conjugates identity to identity, so the `λ`-diagonal is gauge-independent.)
    * **gauge canonicality**, the real risk.
      §3 needs bonds `i` and `i + L` *identically labelled*, and `_fixedpoint_cell` compares entry for entry; a factorisation is defined only up to a phase per column and up to pivot tie-breaking, and pivoting is discontinuous in near-ties.
      Needs a deterministic gauge fix on top (phase of each column's largest entry, a total order on ties) or the cell never converges.
    * `_identity_channels` detects the two boundary vectors as *bare* pass-through backbones; a rotation smears them across columns and they stop being one-hot.
      The channels would have to be pinned out of the rotation and only the complement factorised.
    * the contract shifts from **symbolically exact to tolerance-based** (channel columns differ by factors `λ^k`), and faithfulness would have to move from the `mpo_terms` round-trip to a dense comparison — as `test_irrep_graph.jl` already does for the SVD path.
    Mechanically, `_irrep_graph_svd` also seeds eagerly (`lazy = false`) where channels need the lazy path, and `SVDBondAlgorithm` throws on infinite chains outright.
  * **Jordan blocks** (polynomial × exponential decay), i.e. a channel exiting into another channel.
    That makes the cyclic tails compositional, which is where partition refinement finally earns its keep.
  * **Sum-of-exponentials fitting** for power laws, layered on top of `expterm` (the representation is ready; only the fit is missing).
  * A channel's exit-block tail is interned in the channel id space, so it does not merge with an identical *finite* term's tail — a missed merge, never a wrong answer.
    Sharing one intern table between `_suffix_ids` and the channel names would fix it.
  * A per-*cell* rather than per-site decay convention, i.e. a period-`L` diagonal.
* Inherited from `persistent-graph-mpo.md` §5: fermionic/JW strings, `GenericFusion` multi-channel.
