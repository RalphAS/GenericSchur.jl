# Hessenberg factorization

if VERSION < v"1.3"
    struct HessenbergFactorization{T, S<:StridedMatrix, U} <: Factorization{T}
        data::S
        τ::Vector{U}
    end
    function HessenbergFactorization(A::S, τ::AbstractVector{U}
                                     ) where {S <: AbstractMatrix{T}, U} where T
        HessenbergFactorization{T,S,U}(A, τ)
    end

    Base.size(H::HessenbergFactorization, args...) = size(H.data, args...)

    function _hessenberg!(A::StridedMatrix{T}) where T
        n = LinearAlgebra.checksquare(A)
        τ = Vector{Householder{T}}(undef, n - 1)
        for i = 1:n - 1
            xi = view(A, i + 1:n, i)
            t  = _reflector!(xi)
            H  = Householder{T,typeof(xi)}(view(xi, 2:n - i), t)
            τ[i] = H
            lmul!(H', view(A, i + 1:n, i + 1:n))
            rmul!(view(A, :, i + 1:n), H)
        end
        return HessenbergFactorization{T, typeof(A), eltype(τ)}(A, τ)
    end

    function _materializeQ(H::HessenbergFactorization{T}) where T
        n = size(H.data, 1)
        Z = Matrix{T}(I, n, n)
        for j=n-1:-1:1
            lmul!(H.τ[j], view(Z, j+1:n, j:n))
            Z[1:j-1,j] .= 0
        end
        Z
    end
else # version v1.3+

function _hessenberg!(A::StridedMatrix{T}) where T
    n = LinearAlgebra.checksquare(A)
    τ = Vector{T}(undef, n - 1)
    for i = 1:n - 1
        xi = view(A, i + 1:n, i)
        t  = _reflector!(xi)
        H  = Householder{T,typeof(xi)}(view(A, i+2:n, i), t)
        τ[i] = H.τ
        lmul!(H', view(A, i + 1:n, i + 1:n))
        rmul!(view(A, :, i + 1:n), H)
    end
    return Hessenberg(A, τ)
end

using LinearAlgebra: QRPackedQ
function _materializeQ(H::Hessenberg{T}) where {T}
    A = copy(H.Q.factors)
    n = checksquare(A)
    # shift reflectors one column rightwards
    @inbounds for j in n:-1:2
        A[1:j-1,j] .= zero(T)
        for i in j+1:n
            A[i,j] = A[i,j-1]
        end
    end
    A[2:n,1] .= zero(T)
    A[1,1] = one(T)
    Q1 = QRPackedQ(view(A,2:n,2:n), H.Q.τ)
    A[2:n,2:n] .= Matrix{T}(Q1)
    A
end

end # v1.3+ branch

LinearAlgebra.hessenberg!(A::StridedMatrix) = _hessenberg!(A)
