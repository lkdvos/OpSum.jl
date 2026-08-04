# Aqua quality assurance.
#
# The README has carried an Aqua badge and `test/Project.toml` has carried the dependency since the
# start, but no test ever ran it. This is that test.

using Test
using Aqua
using OpSum

@testset "Aqua" begin
    Aqua.test_all(OpSum)
end
