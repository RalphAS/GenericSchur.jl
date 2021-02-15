module TGST
# routine tests for the Symmetric Tridiagonal eigensolvers in GenericSchur

using LinearAlgebra
using Test
using GenericSchur
using GenericSchur: geigen!, geigvals!

# some interesting examples from the LAPACK papers

function wilkinson(n,::Type{T}) where T
    if mod(n,2) == 1
        nh = (n-1) >>> 1
        D = T.(vcat(nh:-1:1,[0],1:nh))
    else
        D = abs.(T.(collect(-(n-1):2:(n-1)))/2)
    end
    E = fill(T(1),n-1)
    D,E
end

function clement(n,::Type{T}) where T
    D = zeros(T,n)
    E = [sqrt(T(i*(n+1-i))) for i in 1:n-1]
    D,E
end

function legendre(n,::Type{T}) where T
    D = zeros(T,n)
    E = [T(i)/sqrt(T((2i-1)*(2i)+1)) for i in 2:n]
    D,E
end

function laguerre(n,::Type{T}) where T
    D = T.(collect(3:2:2*n+1))
    E = T.(collect(2:n))
    D,E
end

function hermite(n,::Type{T}) where T
    D = zeros(T,n)
    E = [sqrt(T(i)) for i in 1:n-1]
    D,E
end

# LAPACK-style classes
function st_testmat1(n,itype,::Type{T}=Float64; dtype=:normal) where T
    dtype ∈ [:normal, :unif, :uniform] || throw(ArgumentError("incomprehensible dtype"))
    λ = zeros(T,n)
    ulp = eps(real(T))
    kpar = 1/ulp
    if itype == 1
        λ[1] = 1
        for j=2:n
            λ[j] = 1 / T(j)
        end
    elseif itype == 2
        λ[1:n-1] .= 1
        λ[n] = T(1) / n
    elseif itype == 3
        for j=1:n
            λ[j] = 1 / (kpar^((j-1)/(n-1)))
        end
    elseif itype == 4
        for j=1:n
            λ[j] = T(1) - (T(j-1)/T(n-1)) * (1-1/kpar)
        end
    elseif itype == 5
        λ .= exp.(-log(n) * rand(n))
    elseif itype == 6
        if dtype == :normal
            λ .= randn(n)
        else
            λ .= rand(dtype, n)
        end
    elseif itype == 7
        λ[1:n-1] .= (1:n-1)*ulp
        λ[n] = 1
    elseif itype == 8
        λ[1] = ulp
        for j = 2:n-1
            λ[j] = 1 + sqrt(ulp) * j
        end
        λ[n] = 2
    elseif itype == 9
        λ[1] = 1
        for j=2:n
            λ[j] = λ[j-1] + 100 * ulp
        end
    end
    Q,r = qr(randn(n,n))
    Q = T.(Matrix(Q)) # until LinearAlgebra is fixed
    # FIXME: sign adjustment
    A = Q' * diagm(λ) * Q

    F = hessenberg!(A)
    S = F.H
    return S, λ
end
const itypes1 = 1:9
# downselect so subsetting is reasonable
const itypes1a = [1,3,4,5,6]

function st_testmat2(n,itype,::Type{T}=Float64; dtype=:normal) where T <: Real
    dtype ∈ [:normal, :unif, :uniform] || throw(ArgumentError("incomprehensible dtype"))
    λ = zeros(T,n)
    ulp = eps(real(T))
    if itype == 0
        S = SymTridiagonal(zeros(T,n),zeros(T,n-1))
        λ = zeros(T,n)
    elseif itype == 1
        S = SymTridiagonal(ones(T,n),zeros(T,n-1))
        λ = ones(T,n)
    elseif itype == 2
        S = SymTridiagonal(fill(T(2),n),ones(T,n-1))
        λ = nothing
    elseif itype == 3
        S = SymTridiagonal(wilkinson(n,T)...)
        λ = nothing
    elseif itype == 4
        S = SymTridiagonal(clement(n,T)...)
        λ = T.(collect(-n:2:n))
    elseif itype == 5
        S = SymTridiagonal(legendre(n,T)...)
        λ = nothing
    elseif itype == 6
        S = SymTridiagonal(laguerre(n,T)...)
        λ = nothing
    elseif itype == 7
        S = SymTridiagonal(hermite(n,T)...)
        λ = nothing
    elseif itype == 8
        # exercise the scaling logic, all at once
        small = floatmin(T) / ulp
        large = floatmax(T) * ulp
        S1 = SymTridiagonal(wilkinson(n,T)...)
        S = gluemats(small * S1, S1, large * S1)
    else
        throw(ArgumentError("unimplemented itype"))
    end
    S,λ
end
const itypes2 = 1:8
const broken_cases2 = [(BigFloat, 8)]

function gluemats(mat1::SymTridiagonal{T}, mats...; gval=zero(T)) where T
    D = diag(mat1)
    E = diag(mat1,1)
    for mat in mats
        append!(D, diag(mat))
        push!(E, gval)
        append!(E, diag(mat,1))
    end
    SymTridiagonal(D, E)
end

function batch(n, ::Type{T}=Float64; thresh=50, quiet=true) where {T}
    dmax=0.0; vmax=0.0
    for itype in itypes1
        @testset "class 1 type $itype" begin
            A,_ = st_testmat1(n, itype, T)
            de, ve = runtest(diag(A), diag(A,1))
            dmax = max(dmax, de)
            vmax = max(vmax, ve)
            @test de < thresh
            @test ve < thresh
        end
    end
    quiet || println("peak errors: $dmax $vmax")
    dmax=0.0; vmax=0.0
    for itype in itypes2
        @testset "class 2 type $itype" begin
            A,_ = st_testmat2(n, itype, T)
            de, ve = runtest(diag(A), diag(A,1))
            dmax = max(dmax, de)
            vmax = max(vmax, ve)
            if (T, itype) in broken_cases2
                @test_broken de < thresh
            else
                @test de < thresh
            end
            @test ve < thresh
        end
    end
    quiet || println("peak errors: $dmax $vmax")
    nothing
end


maxnorm(A) = maximum(abs.(vec(A)))

function runtest(D::AbstractVector{T}, E; λ=nothing, V=nothing) where T <: Real
    if λ === nothing
        λ, V = geigen!(SymTridiagonal(copy(D), copy(E)))
    end
    n = length(D)
    m = size(V,2)
    ntol = min(m,n)
    myeps = eps(T)
    if m == n
        resnorm = maxnorm(V*Diagonal(λ)*V' - SymTridiagonal(D,E))
    else
        resnorm = maxnorm(Diagonal(λ) - V' * SymTridiagonal(D,E) * V)
    end
    Tnorm = max(opnorm(SymTridiagonal(D,E),1), floatmin(T))
    if resnorm < Tnorm
        d_err =  (resnorm / Tnorm) / (myeps * ntol)
    elseif Tnorm < 1
        d_err = (min(resnorm, Tnorm * ntol) / Tnorm) / (myeps * ntol)
    else
        d_err = min(resnorm / Tnorm, ntol) / (myeps * ntol)
    end
    if m == n
        v_err = maxnorm(V * V' - I) / (myeps * ntol)
    else
        v_err = maxnorm(V' * V - I) / (myeps * ntol)
    end
    d_err, v_err
end

for T in [Float32, Float64, BigFloat]
    @testset "SymTridiagonal $T" begin
        batch(32, T)
    end
end

end # module
