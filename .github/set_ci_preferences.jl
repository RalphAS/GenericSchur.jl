#!/usr/bin/env julia 

open(joinpath(dirname(@__DIR__), "LocalPreferences.toml"), "a") do io
    println(io, """
    [GenericSchur]
    piracy = "$(ARGS[1])"
    """)
end
