# research/

Background notes and algorithm studies that inform OpSum.jl's development.
Not part of the package.

- [persistent-graph-mpo.md](persistent-graph-mpo.md) — the finite-chain sweep design: `ITOGraph`, the
  five-phase `_at_site!` sweep, interned suffix classes and lazy insertion (which together make it
  linear in `N` for finite-range models), and the pending↔started class collision.
- [infinite-mpo.md](infinite-mpo.md) — extending the construction to infinite chains with a repeating
  unit cell: what survives unchanged, why translation covariance forces a canonical ordering, how the
  unit cell is closed, and — in §7 — exponentially decaying terms. §8 collects open follow-ups.
