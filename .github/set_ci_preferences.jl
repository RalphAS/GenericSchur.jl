#!/usr/bin/env julia 

open(joinpath(dirname(@__DIR__), "LocalPreferences.toml"), "a") do io
    val = occursin("no", ARGS[1]) ? "false" : "true"
    println(io, """
    [GenericSchur]
    piracy = "$(val)"
    """)
end
