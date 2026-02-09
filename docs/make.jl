using Documenter, GenericSchur, LinearAlgebra

makedocs(
    modules = [GenericSchur],
    sitename = "GenericSchur.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://RalphAS.github.io/GenericSchur.jl",
        assets=String[],
    ),
    pages = ["Overview" => "index.md",
             "Library" => "library.md"
             ]
)

deploydocs(repo = "github.com/RalphAS/GenericSchur.jl")

