# Persistent-graph MPO construction — design note

*Outcome of the port described in `research/port-handoff.md`: OpSum's symmetry-reduced MPO
construction re-implemented on ITensorMPOConstruction.jl's persistent bipartite-graph + `at_site!`
sweep architecture (see `research/itensor-mpograph-construction.md`), generalized to the non-abelian
(TensorKit `Sector`) ITO machinery.*

New file: `src/operators/irrepgraph.jl` (included between `irreptermtable.jl` and `irrepmpo.jl`).

> **Revised again (stage 4).** The sweep now also reports the two **identity channels** per bond —
> §2.4 — which `src/operators/jordanmpo.jl` uses to emit a Jordan-form MPO. No bond dimension of
> `irrep_mpo` changed: the finish-class forcing that §2.4 describes runs only on the Jordan path, and
> on every reference model it produced the same cover anyway.

> **Revised again (stage 3).** The transient-frontier oracle `_irrep_bipartite` is **deleted** — §3
> explains why it was never able to validate the general case — and the two sweeps have been
> refactored into one skeleton with a pluggable `BondStrategy` (`VertexCover`, `SequentialSVD`,
> `IndependentSVD`; see `src/algorithms.jl`). `_irrep_svd` became `_irrep_independent_svd` and now
> shares the class-interning machinery of §2.1. Both SVD semantics are reachable through
> `SVDBondAlgorithm(trunc; sweep = …)`, closing the §4 follow-up. Every load-bearing `@assert` in the
> sweep is now an explicit `throw`. No bond dimension changed.

> **Revised.** As first written, the sweep was `Θ(N·M)` — quadratic for a finite-range model whose bond
> dimension is `O(1)` — because every term was a live right vertex from bond 1 and each term's full
> length-`N` path was materialised and sorted up front. §2.1 (interned suffix classes) and §2.2 (lazy
> insertion) replace both; §2.3 has the numbers. The output contract, and every bond dimension, are
> unchanged.

## 1. Data structures

```julia
struct LeftVertex{I<:Sector}
    link::Int          # incoming bond index (row into the previous bond's basis)
    key::ITOKey{I}     # on-site ITO key (op, running bond charge, vertex) applied at this site
end                    # ITensor's fermion/JW-string slot is intentionally omitted (no fermions here)

mutable struct ITOGraph{I<:Sector}
    tt::ITOTermTable{I}; N::Int
    # fixed suffix-class machinery (`_suffix_ids`, §2.1):
    K::Int                 # arity(tt)
    sufid::Matrix{Int}     # (K+1)×M interned id of each contiguous column suffix j:K (0 == exhausted)
    # persistent right-vertex state (shrinks via suffix-merge):
    rrepr::Vector{Int}     # right vertex -> representative term id
    rcur::Vector{Int}      # monotone cursor: first column j with sites[j, rrepr] > current site
    rbond::Vector{I}       # running bond charge just past the current site
    # current bipartite graph (bond i-1 -> i), rebuilt each site:
    lefts::Vector{LeftVertex{I}}
    radj::Vector{Vector{Int}}; wadj::Vector{Vector{ComplexF64}}   # adjacency: (right id, scalar weight)
    nlinks::Int            # incoming bond dimension
    # lazy insertion (§2.2):
    lazy::Bool
    firstsite::Vector{Int}                      # term -> first active site (1 for a K=0 term)
    pend_at::Vector{Vector{Int}}                # site -> terms entering there
    pendbysig::Dictionary{Tuple{Int,I},Int}     # pre-start suffix signature -> pending term
    inserted::BitVector; nremaining::Int
    rsent::Int; startleft::Int; startidx::Int   # sentinel / start-channel bookkeeping, per bond
    # per-bond scratch (slot, vlocal, firstleft, remap, siggroups) — see the docstring
end
```

The non-abelian mapping (handoff §"the key idea"): a **right vertex is a suffix class** (identified by a
representative term id; classes enter at their term's first active site and thereafter only merge — §2.2),
and a **left vertex** is
`(incoming link, on-site ITOKey)`. `ITOKey.bond` — the running fusion charge *out of* the site — is
the non-abelian analogue of ITensor's additive QN flux (a fusion *outcome*, not a sum). The graph
sweep itself stays **scalar**: reduced coefficients are `ComplexF64` and the min-vertex-cover / SVD
operate on plain matrices. Non-abelian structure enters only (a) in what makes a bond state distinct
(the augmented `ITOKey` → `bondsectors`) and (b) at tensor assembly (`irrep_mpo_tensors`, unchanged).

## 2. The sweep (`_at_site!`, five phases)

Exactly ITensor's `at_site!` (doc §6), sharing phases 1/2/5 between both graph strategies —
`_at_site!` is the skeleton and `_bond_basis!(g, i, nU, nV, strategy)` is the only thing that
dispatches, returning `(nout, site_dict, secW, nextedges_global)`:

1. **Suffix-merge** (`_suffix_merge!`) — merge right vertices "equal from site `i+1` on", by grouping
   the live ones on an `O(1)` suffix signature (§2.1). `Θ(live_i)` per bond.
2. **Connected components** (`bipartite_connected_components`) — reused verbatim.
3. **Bond basis** (`_bond_basis!`, the strategy plug point) — per-component VC (`_vc_component`)
   or a whole-bond SVD; see §3/§4.
4. **Assemble the bond** — concatenate component ranks (offsets), collecting `secW` charges and the
   forwarded edges.
5. **Build the next graph** (`_build_next_graph!`) — carry the surviving right vertices over; tag
   fresh left vertices with the outgoing bond index `j` as their `link`, bucketed by `op@(i+1)`; and
   inject the terms whose first active site is `i+1` (§2.2).

**Coefficient flow** matches the transient sweep's covered-U / covered-V rule (handoff §3, doc §6):
covered-left forwards its edge weights unchanged and emits the bare letter; covered-right resets the
forwarded weight to 1 and folds `key.op × weight` into the block for every uncovered incident left.
Component bond-charge purity is checked per edge (an explicit `throw`, not an `@assert`: it
guards a silently wrong operator, and `@assert` may be compiled out).

Everything is driven off the sparse adjacency lists: no `nU × nV` coefficient matrix is materialised
anywhere on the VC path (only `_svd_at_site!` densifies, via `_dense_bond_matrix`, since its per-bond
SVD is dense regardless), and `min_vertex_cover_bipartite` takes adjacency lists directly.

### 2.1 Suffix classes without paths

The original port materialised each term's **full length-`N` path**, sorted those lexicographically and
drove the merge off a longest-common-prefix array. That is `Θ(M·N)` memory and `Θ(M·N log M)` time
before the sweep even starts, and it cannot absorb right vertices created mid-sweep — both fatal for
§2.2. It is replaced by interning.

`tt.sites` columns are ascending and zero-padded, so a term's active factors at sites `> i` are always
a *contiguous suffix* `j₀:K` of its column. `_suffix_ids` interns those bottom-up in `Θ(M·K)`:

```julia
sufid[K+1, t] = EMPTY
sufid[j, t]   = tt.sites[j,t] == 0 ? EMPTY : intern((tt.sites[j,t], tt.keys[j,t], sufid[j+1,t]))
```

A suffix *path* is then identified by the two-word signature `(sufid[j₀, t], running bond charge)`.
**Both components are needed.** `sufid` fixes the remaining factor list, and with it the pass-through
charge at every idle site *except* those before the first remaining factor — which carry the charge
accumulated so far, i.e. the second component. `j₀` and the running charge are maintained by a monotone
per-right-vertex cursor (`rcur`/`rbond`) that advances at most `K` times over the whole sweep, so the
signature is `O(1)` per live right vertex per bond. The same cursor yields the next-site key for phase
5, so `_build_next_graph!` no longer calls `_op_at_ito` once per *edge*.

### 2.2 Lazy insertion, and the collision that makes it subtle

Seeding every term eagerly means every term is a live right vertex from bond 1: each one hangs off the
identity/start left vertex, and each bond pays the merge, the remap, the forwarding and the next-graph
bucketing over all of them. That is `Θ(N·M)` — `Θ(N²)` for a finite-range model whose bond dimension is
`O(1)`, which is exactly the scaling bug this note previously recorded as "total merge work is O(N·M)".

So a term's right vertex is created only once it is reachable: at its first active site
(`_build_next_graph!` injects it on the start channel with the term's coefficient as the edge weight —
precisely the weight the eager sweep would have been carrying along that channel since bond 0), and the
terms that have not started yet are represented collectively by **one sentinel right vertex** on the
start left vertex `L₀`. Summed over the sweep the live right vertices then number `Θ(Σ_terms span)`
instead of `Θ(N·M)`.

*Why the sentinel is exact.* Every still-pending term is a **pendant** on `L₀`: its key at site `i` is
the trivial pass-through, its link is the start channel, and its class is a singleton
(`ITOTermTable` dedups identical active content) that matches no live class — otherwise the promotion
below would have fired. Collapsing `k ≥ 1` pendants into one sentinel preserves the minimum vertex
cover: for `k ≥ 2` any cover omitting `L₀` must contain all `k`, and swapping them for `L₀` shrinks it,
so `L₀` is in every minimum cover and the pendants contribute nothing; for `k = 1` both `{L₀}` and
`{pendant}` are size-1 covers. Pendants induce no extra connectivity, so the component partition — and
hence the per-component cover decomposition — is unchanged.

No forcing is needed in the cover either, and in fact something sharper holds: **König's construction
never covers a degree-1 right vertex**, so `L₀` is *always* covered-left. In a maximum matching `L₀` is
matched (otherwise `L₀`–sentinel augments), and the sentinel can be reached by the forward alternating
search only from `L₀` — either along their matching edge, which the search skips, or as a free vertex,
which would complete an augmenting path. So `visitedV[sentinel]` is false, `L₀` is never visited, and the
start channel is always `L₀`'s covered-left index emitting the bare pass-through into `(L₀.link, m₀)`.
That is also why lazy and eager insertion produce *identical* covers rather than merely equal-sized ones.
`_vc_component` keeps the dual branch (an uncovered `L₀` folding `passthrough × 1` into the same block)
as an unreachable fallback, so a future change to the cover construction cannot silently yield a bond
index with no identity channel; it is flagged as such in the code.

*The collision.* `_op_at_ito` fills idle sites with a pass-through carrying the **running** bond charge,
so a *started* term whose accumulated charge has fused back to `unit(I)` is indistinguishable, over its
idle sites, from a term that has not started yet. If its remaining factors then coincide with the whole
content of a pending term, the two suffix classes are **genuinely equal**, and the eager sweep merges
them — covering the shared right vertex instead of spending a bond index. Minimal counterexample
(trivial sector, `N = 3`, `H = couple(A₁,B₃;to=unit) + ½·B₃`): `A₁B₃` and the on-site `B₃` share the
class from site 2 on, so every bond is 1-dimensional; a lazy scheme that simply defers the pending term
until site 3 reports `[2, 2, 1]`.

`_promote_pending!` reproduces the merge: `pendbysig` maps each pending term's *pre-start* signature
(its full factor list at the trivial charge) to that term — injectively, since the factor list fixes the
term — and every bond probes it once per live class, `Θ(live_i)`. Suffix-equality-from-`i+1` is monotone
in `i`, so the probe fires at the earliest colliding bond, which is where the eager sweep merges. (For
the same reason, a "merge schedule" variant that defers a pair's union to
`max(merge_site, first-active-sites)` is **unsound** — deferring loses exactly this merge.)

The condition needs a charge-0 letter or an inner line back to the unit, so it is reachable in the
trivial sector (any `h·Xⱼ` alongside `J·XᵢXⱼ`), in U(1)/fermionic models (an `n̂` chemical potential
against an `n̂ᵢn̂ⱼ` interaction; XXZ plus an `Sᶻ` field), and in SU(2) from `K ≥ 3`. **No showcase model
triggers it** — they are all `K = 2` with no on-site terms — so `test/test_irrep_graph.jl` carries
dedicated cases for all three sectors; see §6.

### 2.3 Cost

Let `M` be the number of terms, `K = arity(tt)`, and a term's *span* the number of bonds between its
first and last active site.

| | preprocessing | sweep |
|---|---|---|
| finite-range (`M = Θ(N)`, `D = O(1)`) | `Θ(M·K)` | `Θ(N)` |
| all-to-all (`M = Θ(N²)`, `D = Θ(N)`) | `Θ(M·K)` | `Θ(N³)` |

Both are `Θ(M·K) + Θ(Σ_terms span)`: each in-flight term contributes one edge at each bond it crosses.
The all-to-all `Θ(N³)` is **intrinsic** to an exact minimum-vertex-cover sweep, not an artefact — at
bond `b` the coefficient matrix `J(n,m)` restricted to `n ≤ b < m` is genuinely dense, so the edges have
to be there. `SVDBondAlgorithm` with truncation is the answer for large long-range systems.

**Deterministic counts are the primary evidence**, because they isolate the mechanism from every
constant factor. Summed over the sweep, live right vertices for a Heisenberg chain were
`135 / 527 / 2079 / 8255` at `N = 16/32/64/128` — exactly `N²/2 + N/2 − 1`, peaking at `M` — and are now
`44 / 92 / 188 / 380`, i.e. `≈ 3N`, with a peak of **3 regardless of `N`**.

Fitted exponents of the `mpo_bipartite` benchmark group (`--sweep full`, OLS in log-log per
`scripts/plot_benchmarks.jl`; see `docs/src/assets/phases.png`):

| | after |
|---|---|
| the ten finite-range models | **`≈ N^1.0 … N^1.15`** |
| `haldane_shastry`, `powerlaw_a3` | **`≈ N^2`** over `N = 8 … 256` — the `Θ(N³)` bound is the asymptote |

Those are rounded deliberately. Repeating the sweep on a shared machine moves the finite-range fits
within `1.0 … 1.14` and the long-range ones over `1.9 … 2.24` (only seven points, and the largest is a
quarter-second), so a three-digit exponent would be false precision. The long-range fit sitting below
its `Θ(N³)` asymptote is expected: the `Θ(M)` class term still outweighs the `Θ(Σ span)` edge term at
these sizes.

Be careful reading a *before* exponent off the old figure: that figure plots the **total** (term-sum
assembly + compression) fitted from `N = 8`, where per-call overhead inflates it, so its `~N^2.5` is not
the compression's exponent. Measuring the parent commit's compression the same way as the table above
gives only `N^1.34 … N^1.50` over the old sweep sizes (`8 … 256`) — the `Θ(N²)` term simply does not
dominate yet at those sizes. Extending the parent to `N = 2048` raises the fit to `N^1.60` with a local
slope of `2.35` across the last octave, which is the honest indication of the quadratic. This is why the
counts above, not a fitted exponent, are what the claim rests on.

End-to-end `irrep_mpo` (term table + sweep), same measurement both sides: Heisenberg `N = 512`
`29.1 ms / 66.2 MB → 3.0 ms / 7.2 MB`; Haldane-Shastry `N = 128` `230 ms / 515 MB → 36 ms / 53 MB`.

Two caveats worth keeping in view. First, `Σ_i E_i` for an all-to-all model is still `Θ(N³)` — its local
exponent rises `2.69 → 2.91` over `N = 16 → 128` — which is the bound above. Second, with the
compression linear the *pipeline* is no longer dominated by it — see §5.

### 2.4 The two identity channels, and Jordan emission

`src/operators/jordanmpo.jl` emits the compressed MPO in Jordan (upper-triangular finite-state
automaton) form, which is what an `MPOHamiltonian` implementation consumes. That needs two named bond
indices at every internal bond: a **start** channel ("nothing placed yet") at position 1 and a
**finish** channel ("everything placed") at position `end`, both carrying `1 · id` on the diagonal.
`_irrep_graph_channels` returns them alongside `(Ws, bondsectors)`; `_at_site!` reads them off the
cover.

*Start.* Already tracked as `g.startidx` — it is `L₀`'s covered-left index, and §2.2 shows `L₀` is
always covered-left when the sentinel exists. It exists exactly while some term still starts to the
right.

*Finish.* The exhausted, trivial-charge suffix class `g.rfinish` (unique after the suffix merge, since
signatures are). Two vertices can carry the meaning: `g.rfinish` itself, and the left vertex
`g.finishleft = (previous finish index, pass-through)`, whose only neighbour is `g.rfinish`. A minimum
cover takes exactly one of the two when both exist — covering both would leave a degree-one vertex
redundant — and either reading emits the bare pass-through with weight 1, because covered-left emits
`key.op` unweighted and covered-right has the uncovered left fold `key.op × 1` in. Weight 1 propagates
down the chain: the first live finish channel is necessarily a covered-right one, which forwards `1`.

*Forcing.* Left to itself the cover covers `g.rfinish` from the *left* whenever it can, via a term's
last factor: when the class and one incident left vertex form an isolated matched pair — what a bond at
which nothing new finishes looks like — König's alternating search visits neither, so the left vertex
is the one that lands in the cover. The resulting index still means "already finished", but it emits
that factor's **letter times the term coefficient**, not the identity, so it cannot be the Jordan
finish channel. `_force_finish!` therefore
puts `g.rfinish` into every cover on the Jordan path (`ITOGraph(...; jordan = true)`) and drops every
left vertex this makes redundant. That grows the cover by at most one and saves exactly the one padded
index it would otherwise cost, so it is never a loss — and empirically it is free: **no reference model
changes a single bond dimension under forcing.** `L₀` is never adjacent to `g.rfinish` (every class it
carries has at least one factor left to place), so the start channel survives forcing.

*Padding.* Where the cover spends no index on a channel — no start channel once every term has entered,
no finish channel before anything has finished — Jordan emission reserves one anyway. That is the whole
of the trade: the emitted MPO is minimal among *Jordan-form* MPOs, and at most `+2` per bond over the
unconstrained minimum `irrep_mpo` returns. On the reference models it is `+1` at the first internal
bond, `+1` at the last, `0` in the bulk, so a nearest-neighbour Heisenberg chain comes out at the
textbook uniform bond dimension 3. Padded channels are dead: nothing enters a padded finish chain (site
1 has no finish row) and a padded start chain reaches nothing (it exists only where no term starts to
its right), so neither can change the operator.

## 3. `VertexCover` — the default `BipartiteAlgorithm` strategy

Per component, `min_vertex_cover_bipartite` chooses the bond basis (covered-left indices first, then
covered-right). This is the wired default: `irrep_mpo(H, sites, BipartiteAlgorithm())` routes here.

**The oracle is gone.** Through stage 2 this sweep was validated differentially against the
transient-frontier `_irrep_bipartite`: same per-sector bond dimensions and same represented operator
on every Hamiltonian in the suite (`U1Irrep`, `SU2Irrep`, `Trivial`; K ∈ {0,1,2,3}; decoupled
multi-component bonds; chains up to N=8). That oracle has now been deleted, for two reasons.

1. It was `Θ(M·N²)` — it re-materialised a suffix path per strand per bond — and was the main reason
   `test_irrep_graph` was the slowest file in the suite.
2. **It was strictly less general than the code it validated.** `_irrep_bipartite` keyed a suffix
   class on `_op_at_ito` alone (`_suffix_path`), which omits the running bond charge, so in the
   non-abelian `K ≥ 2` regime it over-merges two classes that share their remaining factors but
   differ in accumulated charge — and trips its own sector-purity assert. The graph sweep handles
   those (the `suffix signature needs the running bond charge` testset). Consequently
   `reference_hamiltonians()` could never contain such a case, and parity testing alone never
   covered §2.1's charge component.

What replaces it is stronger, because it does not depend on a second implementation: the `mpo_terms`
round-trip (exact, at any N, valid for fermions), the `instantiate` dense oracle through the
assembled `TensorMap`s on the small cases, and — the part a lossless-but-worse compression would slip
past — **pinned per-bond sector profiles**, charge by charge, for every reference Hamiltonian.

## 4. The two SVD strategies

`SequentialSVD` (the ITensor QR-backend port) rides the persistent graph; `IndependentSVD` cannot,
and gets its own pass. Both are selected by `SVDBondAlgorithm(trunc; sweep = …)`, defaulting to
`IndependentSVD` — the historical behaviour.

### 4.1 `SequentialSVD` (`_bond_basis!` on the graph)

Phases 1/2/5 are shared with the VC step; only the basis choice differs: the whole
bond's scalar coefficient matrix is assembled as a **charge-graded** `TensorMap C : Ppre ← Psuf`
(block-diagonal in the bond charge, so `svd_trunc` does the per-sector SVD *and* the global
across-sector truncation at once), the left singular vectors `U` become the compressed bond basis
(block entry `key.op × U[u,m]`), and `R = S·Vᴴ` forwards the coefficient onto the next bond (folded
into the block at the last site).

It seeds with `ITOGraph(tt, N; lazy = false)`: it densifies the bond anyway, so lazy insertion would
buy nothing while adding a sentinel column for the SVD to carry.

### 4.2 `IndependentSVD` (`_irrep_independent_svd`)

Every bond is compressed on the *raw* prefix/suffix classes, independently of what its neighbours
kept — which is precisely why it cannot ride the persistent graph, whose invariant is that bond `b`
is expressed in the basis bond `b-1` left behind. It therefore keeps its own pass, but **not** its
own class-interning code: prefix classes are `_prefix_ids` (the mirror of §2.1's `_suffix_ids`; no
charge component is needed on that side, because the prefix factor list fixes the running charge),
and suffix classes are §2.1's `(sufid, running charge)` signature exactly. Only the current bond's
`Θ(M)` assignment is held at a time, plus one `interned id → dense column` dictionary per bond —
replacing the two dense `M × (N-1)` id matrices the old `_irrep_svd` allocated.

### 4.3 The two truncation semantics

Losslessly the two agree: same per-sector bond dimensions on the internal bonds, same operator. (The
right boundary differs cosmetically — `IndependentSVD` hardcodes it to `unit(I)`, `SequentialSVD`
reports the true total charge.) Under truncation they diverge **by design**:

| | `truncrank(k)` means |
|---|---|
| `IndependentSVD` (default) | `k` indices **per bond** |
| `SequentialSVD` | `k` indices **after upstream truncation** — an aggressive early truncation starves the downstream bonds |

Neither is more correct; `IndependentSVD` remains the default so that `SVDBondAlgorithm()` and
`SVDBondAlgorithm(trunc)` keep meaning what they always did.

## 5. Contract & scope

- Output contract **unchanged**: every strategy returns
  `(Ws::Vector{SparseMatrixCSC{OnsiteOp{I}, Int}}, bondsectors::Vector{Vector{I}})`,
  consumed by `mpo_terms` and `irrep_mpo_tensors` as-is. `ITOKey.vertex` is threaded through
  `LeftVertex.key`, but is `1` throughout the multiplicity-free K ≤ 2 scope, so `bondsectors`
  (charges only) needs no extension.
- Reused verbatim: `bipartite_connected_components`, `_op_at_ito`,
  `bondcharges`/`vertexlabels`/`caterpillar_trees`/`_tree_from_bonds`, `_bond_space`/`_deg_indices`,
  `sparse_from_dict`, `increaseindex!`. `min_vertex_cover_bipartite` gained an adjacency-list method
  (now the primary one; the dense-matrix form survives as a convenience for callers that only
  have an adjacency matrix, and is covered by `test_bipartite.jl`).
- **Follow-ups** (out of scope, as in the handoff): fermionic/JW strings (the omitted `LeftVertex`
  slot) and `GenericFusion` multi-channel (vertex > 1). Wiring the sequential SVD backend is done
  (§4.3).
- **New bottleneck — since fixed.** With the compression linear, symbolic `TermSum` accumulation
  became the dominant cost for finite-range models: building an `N = 8192` Heisenberg chain took
  ~0.83 s against ~0.05 s to compress it, measuring `N^1.3` for `sum([...])` and `N^2.3` for
  `reduce(+, generator)` over `N = 512 … 8192`. That was `TermSum` addition — a
  `Dictionary{TermKey, coeff}` rebuilt from scratch on every `+`, hashing keys that carry a
  `Vector{Int}` of sites, a vector of ops and a `FusionTree` — not the sweep. It has since been
  replaced by an append-only column store whose columns are the `(site, ITOKey)` factors the term
  table already wanted, with the normal form taken once and lazily; assembly is now linear in `M`
  for every accumulation pattern.
- `_irrep_independent_svd` is still `Θ(M·N)` in time (every bond re-classifies every term), which is
  intrinsic to compressing each bond independently. It no longer allocates the two dense
  `M × (N-1)` id matrices.

## 6. Tests

`test/test_irrep_graph.jl` (standalone, discovered by `ParallelTestRunner`):
- `VertexCover sweep — lossless, dense-exact, pinned bond sectors` — for every reference Hamiltonian:
  the `mpo_terms` round-trip, the **pinned per-bond charge profile** (`EXPECTED_BONDS`), and, on the
  N ≤ 3 dim-1-total-charge cases, the assembled tensors contracted against the `instantiate` dense
  oracle. The reference list includes the §2.2 collision in all three sectors (`SU2 K=3 tail collides
  with on-site field`, `U1 charge-0 on-site collides with two-site tail`, `trivial on-site collides
  with two-site tail`) plus the boundary shapes `SU2 no term starts at site 1` and `SU2 first active
  site == N`.
- `the default selector is BipartiteAlgorithm / VertexCover`.
- `SequentialSVD (lossless) reconstructs the operator` — via assembled-tensor contraction vs the dense
  oracle (N ≤ 3, dim-1-total-charge cases).
- `the two lossless SVD sweeps agree on the internal bonds`.
- `K=0 identity terms` — checked via the pinned bond profile + the `mpo_terms` round-trip rather than
  the dense oracle, because `instantiate` cannot represent a `TermSum` mixing K=0 and K>0 terms (a
  pre-existing limitation of the oracle, not of the sweep).
- `pending suffix class can collide with a started one` — the sharp guard for §2.2: pins the minimal
  counterexample's `[1, 1, 1]`, which a naive lazy scheme reports as `[2, 2, 1]`, and contracts the
  result against the dense oracle.
- `incremental suffix-merge on a longer chain` (N=8) and `live right vertices stay bounded
  independently of N` — the peak live right-vertex count is equal at N=5/20/80 and `Σ_i live_i ≤ 4N`.
  (Under lazy insertion the count is *not* monotone any more, since terms enter mid-sweep; the bound is
  the property that buys the linear scaling.)

`test/test_bipartite.jl` covers the cover primitive itself: König's `|cover| == |matching|`, a
brute-force minimum cover on small random graphs, dense/adjacency-method agreement, and a
20 000-deep alternating path (the DFS used to recurse to the alternating-path length).
