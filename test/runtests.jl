using OpSum
using ParallelTestRunner
using ParallelTestRunner: find_tests

# `find_tests` picks up *every* `.jl` file in this directory, so the shared-helper file has to be
# excluded explicitly — it defines functions rather than running tests. Each test file includes it.
testsuite = filter(((name, _),) -> name != "testutils", find_tests(@__DIR__))

runtests(OpSum, ARGS; testsuite)
