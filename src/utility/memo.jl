# Memoisation of the stock bases and operator builders (stockbases.jl), which run a whole `project`
# per operator but are naturally called inside a term loop. Caching by symmetry keeps the caches small
# (a program uses a handful) and needs no invalidation, since the functions are pure and
# `SiteOperator` has no in-place API. The lock guards concurrent building across threads.

const _OPCACHE_LOCK = ReentrantLock()

_cached(f, cache, key) = @lock _OPCACHE_LOCK get!(f, cache, key)
