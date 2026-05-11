using OpSum: OpSum
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(OpSum, :DocTestSetup, :(using OpSum); recursive = true)

makedocs(;
    modules = [OpSum],
    authors = "Lukas Devos",
    sitename = "OpSum.jl",
    format = Documenter.HTML(;
        canonical = "https://lkdvos.github.io/OpSum.jl", edit_link = "main", assets = String[]
    ),
    pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(; repo = "github.com/lkdvos/OpSum.jl", devbranch = "main", push_preview = true)
