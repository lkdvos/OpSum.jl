# Memoisation of the on-site operator builders
# ============================================
# `spin(V)` recomputes a square root and `matrixunit(V, out, in)` runs a whole `project` (a dense
# contraction per candidate letter) — both cheap in isolation, both naturally written *inside* a term
# loop, and both previously documented as "build them once, outside the loop" rather than fixed.
#
# Caching makes that advice unnecessary. The keys are spaces and sectors, of which a program uses a
# handful, so the caches do not grow with the system size and never need invalidating: the functions
# are pure, and a `SiteOperator` has no in-place API — every one of its arithmetic operations copies. The
# lock is there because nothing stops a caller from building terms on several threads.

const _OPCACHE_LOCK = ReentrantLock()

_cached(f, cache, key) = @lock _OPCACHE_LOCK get!(f, cache, key)
