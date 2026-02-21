module GSTest
using GenericSchur
using GenericSchur: geigvecs
using Test
using Random
using LinearAlgebra
using LinearAlgebra.BLAS: BlasFloat

include("TMGlib.jl")
using .TMGlib

using Aqua

const piracy = GenericSchur._get_piracy()

Aqua.test_all(GenericSchur; piracies = !piracy)

_vbst = parse(Int, get(ENV, "TEST_VERBOSITY", "0"))
const verbosity = Ref(_vbst)

if parse(Int, get(ENV, "TEST_RESEEDRNG", "0")) != 0
    let seed = round(Int, 1024 * rand(RandomDevice()))
        @info "rng seed is $seed"
        Random.seed!(seed)
    end
end

include("testfuncs.jl")

# temporarily up front
include("generalized.jl")
include("gordschur.jl")

include("complex.jl")

if piracy
    include("wrappers.jl")
# else
    # should be handled by Aqua
end

include("symtridiag.jl")
include("balance.jl")
include("real.jl")
#include("complex.jl")

include("ordschur.jl")

# include("generalized.jl")
# include("gordschur.jl")

include("errors.jl")

end # module
