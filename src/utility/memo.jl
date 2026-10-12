# Memoisation of the stock bases and builders (each runs a whole `project`). Keyed by symmetry, so
# the caches stay small and need no invalidation (pure functions). The lock guards threads.

const _OPCACHE_LOCK = ReentrantLock()

_cached(f, cache, key) = @lock _OPCACHE_LOCK get!(f, cache, key)
