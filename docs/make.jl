using OpSum: OpSum
using Documenter: Documenter, DocMeta, deploydocs, makedocs

DocMeta.setdocmeta!(OpSum, :DocTestSetup, :(using OpSum); recursive=true)

include("make_index.jl")

makedocs(;
    modules=[OpSum],
    authors="ITensor developers <support@itensor.org> and contributors",
    sitename="OpSum.jl",
    format=Documenter.HTML(;
        canonical="https://ITensor.github.io/OpSum.jl", edit_link="main", assets=String[]
    ),
    pages=["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(; repo="github.com/ITensor/OpSum.jl", devbranch="main", push_preview=true)
