# Hessenberg factorization

function _hessenberg!(A::StridedMatrix{T}) where T
    n = LinearAlgebra.checksquare(A)
    τ = Vector{T}(undef, n - 1)
    xwrk = Vector{T}(undef, n)
    for i = 1:n - 1
        xi = view(A, i + 1:n, i)
        t  = _reflector!(xi)
        H  = Householder{T,typeof(xi)}(view(A, i+2:n, i), t)
        τ[i] = H.τ
        lmul!(H', view(A, i + 1:n, i + 1:n))
        # rmul!(view(A, :, i + 1:n), H)
        rmul!(view(A, :, i + 1:n), H, xwrk)
    end
    return Hessenberg(A, τ)
end

    function _hessenberg!(Ah::Union{Hermitian{T}, Symmetric{T}}) where T <: Real
        _hehessenberg!(Hermitian(Ah), Val(Symbol(Ah.uplo)))
    end
    function _hessenberg!(Ah::Hermitian{T})  where T <: Complex
        _hehessenberg!(Ah, Val(Symbol(Ah.uplo)))
    end

    function _hehessenberg!(Ah::Hermitian{T}, ::Val{:L}) where T
        # based on LAPACK zhetd2
        A = Ah.data
        n = LinearAlgebra.checksquare(A)
        τ = Vector{T}(undef, n - 1)
        d = zeros(real(T), n)
        e = zeros(real(T), n-1)
        A[1,1] = real(A[1,1])
        for i = 1:n - 1
            # reflector to annihilate A[i+2:n,i]
            ξ = view(A, i+1:n, i)
            τi  = _reflector!(ξ)
            e[i] = real(A[i+1,i])
            if !iszero(τi)
                ξ[1] = one(T)
                # store x = τᵢ A ξ in tail of τ
                w = view(τ,i:n-1)
                mul!(w, Hermitian(view(A, i+1:n, i+1:n),:L), ξ, τi, false)
                # compute w = ξ - (τᵢ/2) (xᴴ ξ) ξ
                α = - (τi / 2) * dot(w, ξ)
                w .= w .+ α .* ξ
                # apply transformation as rank-2 update
                rank2update!(view(A, i+1:n, i+1:n), -1, ξ, w, Val(:L))
            else
                A[i+1, i+1] = real(A[i+1, i+1])
            end
            A[i+1, i] = e[i]
            d[i] = real(A[i,i])
            τ[i] = τi
        end
        d[n] = real(A[n,n])
        return Hessenberg(A, τ, SymTridiagonal(d, e), Ah.uplo)
    end

    function _hehessenberg!(Ah::Hermitian{T}, ::Val{:U}) where T
        # based on LAPACK zhetd2
        A = Ah.data
        n = LinearAlgebra.checksquare(A)
        τ = Vector{T}(undef, n - 1)
        d = zeros(real(T), n)
        e = zeros(real(T), n-1)
        A[n,n] = real(A[n,n])
        xi = zeros(T,n)
        for i=n-1:-1:1
            # reflector to annihilate A[1:i-1,i+1]
            xi[1] = A[i,i+1]
            xi[2:i] .= A[1:i-1,i+1]
            ξ = view(xi,1:i)
            τi = _reflector!(ξ)
            e[i] = real(xi[1])
            A[1:i-1,i+1] .= xi[2:i]
            if !iszero(τi)
                A[i,i+1] = one(T)
                ξ = view(A, 1:i, i+1)
                # store x = τᵢ A ξ in head of τ
                w = view(τ,1:i)
                mul!(w, Hermitian(view(A, 1:i, 1:i),:U), ξ, τi, false)
                # compute w = ξ - (τᵢ/2) (xᴴ ξ) ξ
                α = - (τi / 2) * dot(w, ξ)
                w .= w .+ α .* ξ
                # apply transformation as rank-2 update
                rank2update!(view(A, 1:i, 1:i), -1, ξ, w, Val(:U))
            else
                A[i, i] = real(A[i, i])
            end
            A[i, i+1] = e[i]
            d[i+1] = real(A[i+1,i+1])
            τ[i] = τi
        end
        d[1] = real(A[1,1])
        return Hessenberg(A, τ, SymTridiagonal(d, e), Ah.uplo)
    end

    function rank2update!(A::StridedMatrix, α::Number, x::AbstractVector, y::AbstractVector, ::Val{:U})
        n = length(x)
        for j=1:n
            if iszero(x[j]) && iszero(y[j])
                A[j,j] = real(A[j,j])
            else
                t1 = α * conj(y[j])
                t2 = conj(α * x[j])
                for i=1:j-1
                    A[i,j] += x[i] * t1 + y[i] * t2
                end
                A[j,j] = real(A[j,j]) + real(x[j] * t1 + y[j] * t2)
            end
        end
    end
    function rank2update!(A::StridedMatrix, α::Number, x::AbstractVector, y::AbstractVector, ::Val{:L})
        n = length(x)
        for j=1:n
            if iszero(x[j]) && iszero(y[j])
                A[j,j] = real(A[j,j])
            else
                t1 = α * conj(y[j])
                t2 = conj(α * x[j])
                A[j,j] = real(A[j,j]) + real(x[j] * t1 + y[j] * t2)
                for i=j+1:n
                    A[i,j] += x[i] * t1 + y[i] * t2
                end
            end
        end
    end

using LinearAlgebra: QRPackedQ
function _materializeQ(H::Hessenberg{T}) where {T}
    if H.uplo == 'L'
        return _materializeQ(H, Val(:L))
    else
        return _materializeQ(H, Val(:U))
    end
end

function _materializeQ!(Q::AbstractMatrix, H::Hessenberg{T}, Zwrk) where {T}
    if H.uplo == 'L'
        return _materializeQ!(Q, H, Val(:L), Zwrk)
    else
        return _materializeQ!(Q, H, Val(:U), Zwrk)
    end
end

function _materializeQ(H::Hessenberg{T}, ::Val{:L}) where {T}
    H.uplo == 'L' || throw(ArgumentError("only implemented for uplo='L'"))
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

function _materializeQ!(Q, H::Hessenberg{T}, ::Val{:L}, Atmp) where {T}
    H.uplo == 'L' || throw(ArgumentError("only implemented for uplo='L'"))
    if Atmp != nothing
        A = Atmp
    else
        A = similar(H.Q.factors)
    end
    copyto!(A, H.Q.factors)
    n = checksquare(A)
    nq = checksquare(Q)
    nq == n || throw(ArgumentError("Q must have same size as factors in H"))
    # shift reflectors one column rightwards
    @inbounds for j in n:-1:2
        A[1:j-1,j] .= zero(T)
        for i in j+1:n
            A[i,j] = A[i,j-1]
        end
    end
    Q[2:n,1] .= zero(T)
    Q[1,1] = one(T)
    Q1 = QRPackedQ(view(A,2:n,2:n), H.Q.τ)
    Q[2:n,2:n] .= zero(T)
    @inbounds for j in 2:n
        Q[j,j] = one(T)
    end
    lmul!(Q1, view(Q,2:n,2:n))
    Q
end

# alas, stdlib has no QLPackedQ
function _materializeQ(H::Hessenberg{T},::Val{:U}) where {T}
    H.uplo == 'U' || throw(ArgumentError("only implemented for uplo='U'"))
    A = copy(H.Q.factors)
    n = checksquare(A)
    # shift reflectors one column leftwards
    @inbounds for j in 1:n-1
        for i in 1:j-1
            A[i,j] = A[i,j+1]
        end
        A[j:n,j] .= zero(T)
    end
    A[1:n-1,n] .= zero(T)
    A[n,n] = one(T)
    w = zeros(T,n-1)
    for i in 1:n-1
        A[i,i] = one(T)
        τi = H.τ[i]
        v = view(A,1:i,i)
        # w = A[1:i,1:i-1]' * v
        for j in 1:i-1
            w[j] = dot(view(A, 1:i, j), v)
        end
        # A[1:i,1:i-1] .-= τi * v * w'
        for j in 1:i-1
            t = τi * w[j]'
            for k in 1:i
                A[k,j] -= v[k] * t
            end
        end
        A[1:i-1, i] *= -τi
        A[i,i] = one(T) - τi
        A[i+1:n,i] .= zero(T)
    end
    A
end

LinearAlgebra.hessenberg!(A::StridedMatrix{<: STypes}) = _hessenberg!(A)

using LinearAlgebra: RealHermSymComplexHerm
# this should be S <: StridedMatrix, but for stdlib foolishness
function LinearAlgebra.hessenberg!(Ah::RealHermSymComplexHerm{<: STypes, <: AbstractMatrix})
    _hehessenberg!(Hermitian(Ah), Val(Symbol(Ah.uplo)))
end

