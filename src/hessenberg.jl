
# Hessenberg Matrix
struct HessenbergMatrix{T,S<:StridedMatrix} <: AbstractMatrix{T}
    data::S
end

Base.copy(H::HessenbergMatrix{T,S}) where {T,S} = HessenbergMatrix{T,S}(copy(H.data))

Base.getindex(H::HessenbergMatrix{T,S}, i::Integer, j::Integer) where {T,S} = i > j + 1 ? zero(T) : H.data[i,j]

Base.size(H::HessenbergMatrix) = size(H.data)
Base.size(H::HessenbergMatrix, i::Integer) = size(H.data, i)

function LinearAlgebra.ldiv!(H::HessenbergMatrix, B::AbstractVecOrMat)
    n = size(H, 1)
    Hd = H.data
    for i = 1:n-1
        G, _ = givens!(Hd, i, i+1, i)
        lmul!(G, view(Hd, 1:n, i+1:n))
        lmul!(G, B)
    end
    ldiv!(Triangular(Hd, :U), B)
end

# Hessenberg factorization
struct HessenbergFactorization{T, S<:StridedMatrix,U} <: Factorization{T}
    data::S
    τ::Vector{U}
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
    return HessenbergFactorization{T, typeof(A), eltype(τ)}(A, τ)
end
LinearAlgebra.hessenberg!(A::StridedMatrix) = _hessenberg!(A)

Base.size(H::HessenbergFactorization, args...) = size(H.data, args...)
