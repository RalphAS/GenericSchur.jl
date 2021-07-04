module TGeig
# routine tests for wrappers
# This section is just to check existence and sanity of methods.
# Serious accuracy verification is elsewhere.

using LinearAlgebra
using LinearAlgebra: sorteig!
using Test
using GenericSchur

# allow for different phase factors and perhaps ordering
function _chkeigvecs(v1,v2::AbstractMatrix{T}, isnrml, sameorder=true) where T
    m,n = size(v2)
    if isnrml
        @test norm(v2' * v2 - I) < 50 * n * eps(real(T))
    end
    tmat = v1' * v2
    zT = zero(real(T))
    nok = 0
    for j in 1:m
        # largest and next-largest projections
        if sameorder
            p1 = abs(tmat[j,j])
            tmat[j,j] = 0
            p2 = maximum(abs.(tmat[:,j]))
        else
            p1,p2 = zT,zT
            for i in 1:m
                ta = abs(tmat[i,j])
                if ta > p1
                    p1 = ta
                elseif ta > p2
                    p2 = ta
                end
            end
        end
        ok = (abs(p1-1) < sqrt(eps(real(T))))
        if isnrml
            ok &= (abs(p2) < sqrt(eps(real(T))))
        end
        nok += ok
    end
    @test nok == m
end

let T = BigFloat
    n = 10
    # FIXME: should protect against accidental poor condition
    A = rand(T,n,n)
    for (w,f) in zip([:bare, :hermitian, :symmetric],[identity, Hermitian, Symmetric])
        @testset "wrappers $w $T" begin
            Awrk = f(A)
            E = eigen(Awrk)
            @test norm(Awrk*E.vectors - E.vectors * Diagonal(E.values)) < sqrt(eps(T))
            v = eigvecs(Awrk)
            _chkeigvecs(E.vectors, v, w != :bare)
            λ = eigvals(Awrk)
            @test sorteig!(λ) ≈ sorteig!(E.values)
        end
    end
end

let T = Complex{BigFloat}
    n = 10
    # FIXME: should protect against accidental poor condition
    A = rand(T,n,n)
    for (w,f) in zip([:bare, :hermitian],[identity, Hermitian])
        @testset "wrappers $w $T" begin
            Awrk = f(A)
            E = eigen(Awrk)
            @test norm(Awrk*E.vectors - E.vectors * Diagonal(E.values)) < sqrt(eps(real(T)))
            v = eigvecs(Awrk)
            _chkeigvecs(E.vectors, v, w != :bare)
            λ = eigvals(Awrk)
            @test sorteig!(λ) ≈ sorteig!(E.values)
        end
    end
end

# CHECKME: we should not need to be so lenient,
# but this is sufficient for legitimate applications.
function _chkrcond(x,y,tiny=1e3*eps(eltype(x)))
    ok = true
    for (xi,yi) in zip(x,y)
        x1 = clamp(xi,0,1)
        y1 = clamp(yi,0,1)
        if abs(x1 - y1) > 0.5 * abs(y1) + tiny
            ok = false
        end
    end
    return ok
end

if VERSION > v"1.7.0-DEV.976"
    for T in (Complex{BigFloat},)
        Tref = ComplexF64
        @testset "extended eigen() $T" begin
            n = 10
            Aref = rand(Tref,n,n)
            A = T.(Aref)
            old = precision(real(T))
            @assert old >= 53
            Eref = eigen(Aref, jvl=true, jce=true, jcv=true)
            setprecision(real(T), 53) do
                E = eigen(A, jvl=true, jce=true, jcv=true)
                @test E.values ≈ Eref.values
                @test _chkrcond(E.rconde, Eref.rconde)
                @test _chkrcond(E.rcondv, Eref.rcondv)
                _chkeigvecs(Eref.vectors, E.vectors, false)
                _chkeigvecs(Eref.vectorsl, E.vectorsl, false)
            end
        end
    end
end

end # module
