# Memoisation of pure functions (stock bases, adjoints); the lock guards threads.

const _OPCACHE_LOCK = ReentrantLock()

_cached(f, cache, key) = @lock _OPCACHE_LOCK get!(f, cache, key)
