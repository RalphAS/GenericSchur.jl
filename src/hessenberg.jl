# Hessenberg factorization

if VERSION < v"1.3"
    struct HessenbergFactorization{T, S<:StridedMatrix, U} <: Factorization{T}
        data::S
        τ::Vector{U}
    end
    Base.size(H::HessenbergFactorization, args...) = size(H.data, args...)
end

function _hessenberg!(A::StridedMatrix{T}) where T
    n = LinearAlgebra.checksquare(A)
    τ = Vector{Householder{T}}(undef, n - 1)
    for i = 1:n - 1
        xi = view(A, i + 1:n, i)
        t  = LinearAlgebra.reflector!(xi)
        H  = Householder{T,typeof(xi)}(view(xi, 2:n - i), t)
        τ[i] = H
        lmul!(H', view(A, i + 1:n, i + 1:n))
        rmul!(view(A, :, i + 1:n), H)
    end
    if VERSION < v"1.3"
        return HessenbergFactorization{T, typeof(A), eltype(τ)}(A, τ)
    else
        return Hessenberg(A, τ)
    end
end
LinearAlgebra.hessenberg!(A::StridedMatrix) = _hessenberg!(A)
