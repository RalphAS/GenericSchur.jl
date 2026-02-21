cplxord = t -> (real(t), imag(t))
csort(v) = sort(v, by = cplxord)

# isolate eigvec tests for use with special cases
function vectest(A::Matrix{T}, S::Schur{T2}, vtol; normal = false) where {T, T2}
    n = size(A, 1)
    ulp = eps(real(T))
    tiny = floatmin(real(T))
    if norm(A) < 16 * n * tiny / ulp
        # The numerical analysis in the eigenvector code (LAPACK and ours)
        # is shaky for very small numbers.
        # Normally this is obviated by balancing.
        fac = 1 / sqrt(tiny)
        Ss = LinearAlgebra.Schur(fac * S.T, S.Z, fac * S.values)
        VR = geigvecs(Ss)
        VL = geigvecs(Ss, left = true)
    else
        VR = geigvecs(S)
        VL = geigvecs(S, left = true)
    end
    if verbosity[] > 1
        vec_err = norm(A * VR - VR * diagm(0 => S.values)) / (n * norm(A) * ulp)
        println("r.eigenvector error: $vec_err, vtol = $vtol")
        @test vec_err < 1000
    else
        @test norm(A * VR - VR * diagm(0 => S.values)) / (n * norm(A) * ulp) < vtol
    end
    if verbosity[] > 1
        vec_err = norm(A' * VL - VL * diagm(0 => conj.(S.values))) / (n * norm(A) * ulp)
        println("l.eigenvector error: $vec_err, vtol = $vtol")
        @test vec_err < 1000
    else
        @test norm(A' * VL - VL * diagm(0 => conj.(S.values))) / (n * norm(A) * ulp) < vtol
    end
    if normal
        # check orthonormality of vectors where appropriate
        if verbosity[] > 1 && real(T) == Float16
            trt = norm(VR * VR' - I) / (n * ulp)
            tlt = norm(VL * VL' - I) / (n * ulp)
            @info "Float16 vec orth errors for n=$n: $tlt, $trt"
        end
        if (T == ComplexF16) && (n > 10)
            # is this simply too many ops for this type?
            @test_broken norm(VR * VR' - I) / (n * ulp) < vtol
            @test_broken norm(VL * VL' - I) / (n * ulp) < vtol
        else
            # CHECKME: real ones are not as good for some reason
            votol = (T <: Complex) ? vtol : (T == Float16 ? 10 : 5) * vtol
            @test norm(VR * VR' - I) / (n * ulp) < votol
            @test norm(VL * VL' - I) / (n * ulp) < votol
        end
    end
    return
end

"""
generate a random unitary matrix

Unitary matrices are normal, so Schur decomposition is diagonal for them.
(This is not only another test, but sometimes actually useful.)
"""
function randu(::Type{T}, n) where {T <: Complex}
    if real(T) ∈ [Float16, Float32, Float64]
        A = randn(T, n, n)
        F = qr(A)
        return Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
    else
        # don't have normal deviates for other types, but need appropriate QR
        A = randn(ComplexF64, n, n)
        F = qr(convert.(T, A))
        return Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
    end
end

# random orthogonal matrix
function rando(::Type{T}, n) where {T <: Real}
    if T ∈ [Float16, Float32, Float64]
        A = randn(T, n, n)
        F = qr(A)
        return Matrix(F.Q)
    else
        # don't have normal deviates for other types, but need appropriate QR
        A = randn(Float64, n, n)
        F = qr(convert.(T, A))
        return Matrix(F.Q)
    end
end

"""
Generate Godunov's strongly non-normal matrix with known eigenvalues.

One needs to retain at least 70 bits of precision to have any confidence in
computing the eigenvalues here.
"""
function godunov(T)
    A = convert.(
        T, [
            289 2064 336 128 80 32 16;
            1152 30 1312 512 288 128 32;
            -29 -2000 756 384 1008 224 48;
            512 128 640 0 640 512 128;
            1053 2256 -504 -384 -756 800 208;
            -287 -16 1712 -128 1968 -30 2032;
            -2176 -287 -1565 -512 -541 -1152 -289
        ]
    )
    vals = [-4, -2, -1, 0, 1, 2, 4]
    econd = 7.0e16 # condition of the worst ones
    return A, vals, econd
end
