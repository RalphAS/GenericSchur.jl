module GSTest
using GenericSchur
using Test
using Random
using LinearAlgebra
include("TMGlib.jl")
using .TMGlib

import LinearAlgebra.BLAS.BlasFloat

const verbosity = Ref(0)

cplxord = t -> (real(t), imag(t))
csort(v) = sort(v, by = cplxord)

"""
Generate Godunov's strongly non-normal matrix with known eigenvalues.

One needs to retain at least 70 bits of precision to have any confidence in
computing the eigenvalues here.
"""
function godunov(T)
    A = convert.(T,[289 2064 336 128 80 32 16;
                    1152 30 1312 512 288 128 32;
                    -29 -2000 756 384 1008 224 48;
                    512 128 640 0 640 512 128;
                    1053 2256 -504 -384 -756 800 208;
                    -287 -16 1712 -128 1968 -30 2032;
                    -2176 -287 -1565 -512 -541 -1152 -289])
    vals = [-4,-2,-1,0,1,2,4]
    econd = 7e16 # condition of the worst ones
    A,vals,econd
end

include("wrappers.jl")
include("symtridiag.jl")
include("complex.jl")
include("balance.jl")
include("real.jl")
#include("complex.jl")

include("ordschur.jl")

include("generalized.jl")
include("gordschur.jl")

include("errors.jl")

end # module
