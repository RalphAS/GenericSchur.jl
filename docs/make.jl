using Documenter, GenericSchur, LinearAlgebra

makedocs(
    modules = [GenericSchur],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == true),
    sitename = "GenericSchur.jl",
    pages = ["Overview" => "index.md",
             "Library" => "library.md"
             ]
)

# or maybe just the pkg site?
deploydocs(repo = "github.com/RalphAS/GenericSchur.jl.git")

