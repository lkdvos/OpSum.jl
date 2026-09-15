# Memoisation of the on-site operator builders. `spin(V)` recomputes a square root and
# `matrixunit(V, out, in)` runs a whole `project` (a dense contraction per candidate letter) — both
# cheap in isolation but naturally called inside a term loop. Caching by space/sector keeps the caches
# small (a program uses a handful) and needs no invalidation, since both functions are pure and
# `SiteOperator` has no in-place API. The lock guards concurrent building across threads.

const _OPCACHE_LOCK = ReentrantLock()

_cached(f, cache, key) = @lock _OPCACHE_LOCK get!(f, cache, key)
