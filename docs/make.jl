using OpSum: OpSum
using Documenter: Documenter, DocMeta, deploydocs, makedocs
using Literate: Literate

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const SRC_DIR = joinpath(@__DIR__, "src")

# Example pages, in reading order. Each file's `# # Title` header becomes the page title.
const EXAMPLES = [
    "common.jl",
    "spin_chains.jl",
    "long_range.jl",
    "ladders_and_cylinders.jl",
    "multibody.jl",
    "fermions.jl",
]

# `examples/index.jl` is the source for the docs landing page. The root `README.md` is maintained
# by hand and is not generated from it.
Literate.markdown(
    joinpath(EXAMPLES_DIR, "index.jl"), SRC_DIR;
    flavor = Literate.DocumenterFlavor()
)

for file in EXAMPLES
    Literate.markdown(
        joinpath(EXAMPLES_DIR, file), joinpath(SRC_DIR, "examples");
        flavor = Literate.DocumenterFlavor()
    )
end

DocMeta.setdocmeta!(OpSum, :DocTestSetup, :(using OpSum); recursive = true)

makedocs(;
    modules = [OpSum],
    authors = "Lukas Devos",
    sitename = "OpSum.jl",
    format = Documenter.HTML(;
        canonical = "https://lkdvos.github.io/OpSum.jl",
        edit_link = "main",
        assets = String[],
        size_threshold = 500 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => ["examples/$(first(splitext(f))).md" for f in EXAMPLES],
        "Reference" => "reference.md",
    ],
)

deploydocs(; repo = "github.com/lkdvos/OpSum.jl", devbranch = "main", push_preview = true)
