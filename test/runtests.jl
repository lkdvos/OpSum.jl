using OpSum
using ParallelTestRunner
using ParallelTestRunner: find_tests

# `find_tests` picks up every `.jl` file here, so the shared-helper file is excluded explicitly.
testsuite = filter(((name, _),) -> name != "testutils", find_tests(@__DIR__))

# Most of a test file's time is compilation, and every file exceeds the default worker RSS cap
# (2 GB on CI's macOS runners), which restarts the worker and recompiles everything. Budget 4 GB per
# worker instead, so workers are reused and fewer of them run at once.
const RSS = 4 * Int64(2)^30
runtests(OpSum, ARGS; testsuite, max_worker_rss = RSS, memory_per_worker = RSS)
