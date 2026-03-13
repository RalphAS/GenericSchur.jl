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

_vbst = parse(Int, get(ENV, "TEST_VERBOSITY", "0"))
const verbosity = Ref(_vbst)

env_seed = parse(Int, get(ENV, "TEST_RESEED_RNG", "-1"))
if env_seed >= 0
    let seed = (env_seed > 0) ? env_seed : round(Int, 1024 * rand(RandomDevice()))
        @info "rng seed is $seed"
        Random.seed!(seed)
    end
end

include("testfuncs.jl")

@testset "real nonsymmetric EP" verbose = true include("real.jl")

@testset "complex nonsymmetric EP" verbose = true include("complex.jl")

@testset "generalized EP" verbose = true include("generalized.jl")

@testset "Hermitian EP" verbose = true include("symtridiag.jl")
@testset "balancing" verbose = true include("balance.jl")

@testset "reordering Schur" verbose = true include("ordschur.jl")
@testset "reordering generalized Schur" verbose = true include("gordschur.jl")

@testset "error handling" verbose = true include("errors.jl")

@testset "wrappers" verbose = true include("wrappers.jl")

@testset "Aqua" verbose = true Aqua.test_all(GenericSchur; piracies = !piracy)

end # module
