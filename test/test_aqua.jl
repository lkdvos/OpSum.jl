# Aqua quality assurance.
#
# The README has carried an Aqua badge and `test/Project.toml` has carried the dependency since the
# start, but no test ever ran it. This is that test.

using Test
using Aqua
using OpSum

@testset "Aqua" begin
    # `unbound_args` is marked broken, not skipped: the only offender is the `LocalOp` constructor
    # that `LightSumTypes.@sumtype` generates, and the symbolic tower it belongs to is slated for
    # deletion. Marking it broken means this test starts *failing* once that lands, which is the
    # reminder to flip it back on rather than leave a permanent exemption.
    Aqua.test_all(OpSum; unbound_args = (broken = true,))
end
