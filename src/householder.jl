import Base: *, eltype, size
import LinearAlgebra: adjoint, mul!, rmul!, lmul!

# The reflector! code in stdlib has no underflow or accuracy protection

# These are translations of xLARFG from LAPACK
# LAPACK Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
function _reflector!(x::AbstractVector{T}) where {T<:Real}
    require_one_based_indexing(x)
    n = length(x)
    n <= 1 && return zero(T)
    sfmin = 2floatmin(T) / eps(T)
    @inbounds begin
        α = x[1]
        xnorm = _norm2(view(x,2:n))
        if iszero(xnorm)
            return zero(T)
        end
        β = -copysign(hypot(α, xnorm), α)
        kount = 0
        smallβ = abs(β) < sfmin
        if smallβ
            # recompute xnorm and β if needed for accuracy
            rsfmin = one(T) / sfmin
            while smallβ
                kount += 1
                for j in 2:n
                    x[j] *= rsfmin
                end
                β *= rsfmin
                α *= rsfmin
                # CHECKME: is 20 adequate for BigFloat?
                smallβ = (abs(β) < sfmin) && (kount < 20)
            end
            # now β ∈ [sfmin,1]
            xnorm = _norm2(view(x,2:n))
            β = -copysign(hypot(α, xnorm), α)
        end
        τ = (β - α) / β
        t = one(T) / (α - β)
        for j in 2:n
            x[j] *= t
        end
        for j in 1:kount
            β *= sfmin
        end
        x[1] = β
    end
    return τ
end

function _reflector!(x::AbstractVector{T}) where {T<:Complex}
    require_one_based_indexing(x)
    n = length(x)
    # we need to make subdiagonals real so the n=1 case is nontrivial for complex eltype
    n < 1 && return zero(T)
    RT = real(T)
    sfmin = floatmin(RT) / eps(RT)
    @inbounds begin
        α = x[1]
        αr, αi = reim(α)
        xnorm = _norm2(view(x,2:n))
        if iszero(xnorm) && iszero(αi)
            return zero(T)
        end
        β = -copysign(_hypot3(αr, αi, xnorm), αr)
        kount = 0
        smallβ = abs(β) < sfmin
        if smallβ
            # recompute xnorm and β if needed for accuracy
            rsfmin = one(real(T)) / sfmin
            while smallβ
                kount += 1
                for j in 2:n
                    x[j] *= rsfmin
                end
                β *= rsfmin
                αr *= rsfmin
                αi *= rsfmin
                smallβ = (abs(β) < sfmin) && (kount < 20)
            end
            # now β ∈ [sfmin,1]
            xnorm = _norm2(view(x,2:n))
            α = complex(αr, αi)
            β = -copysign(_hypot3(αr, αi, xnorm), αr)
        end
        τ = complex((β - αr) / β, -αi / β)
        t = one(T) / (α - β)
        for j in 2:n
            x[j] *= t
        end
        for j in 1:kount
            β *= sfmin
        end
        x[1] = β
    end
    return τ
end

# copied from Andreas Noack's GenericLinearAlgebra.jl, with trivial mods

"""
a Householder reflection represented as the essential part of the
vector and the normalizing factor
"""
struct Householder{T,S<:StridedVector}
    v::S
    τ::T
end

# warning; size -> length(v) in GLA, but this makes more sense to us:
size(H::Householder) = (length(H.v)+1, length(H.v)+1)
size(H::Householder, i::Integer) = i <= 2 ? length(H.v)+1 : 1

eltype(H::Householder{T})      where T = T

adjoint(H::Householder{T})      where {T} = Adjoint{T,typeof(H)}(H)

function lmul!(H::Householder, A::StridedMatrix)
    m, n = size(A)
    size(H,1) == m || throw(DimensionMismatch(""))
    v = view(H.v, 1:m - 1)
    τ = H.τ
    for j = 1:n
        va = A[1,j]
        Aj = view(A, 2:m, j)
        va += dot(v, Aj)
        va = τ*va
        A[1,j] -= va
        axpy!(-va, v, Aj)
    end
    A
end

# x is workspace which one can preallocate
function rmul!(A::StridedMatrix{T}, H::Householder,
               x=Vector{T}(undef,size(A,1))
               ) where T
    m, n = size(A)
    size(H,1) == n || throw(DimensionMismatch(""))
    v = view(H.v, :)
    τ = H.τ
    a1 = view(A, :, 1)
    A1 = view(A, :, 2:n)
    mul!(x, A1, v)
    axpy!(one(τ), a1, x)
    axpy!(-τ, x, a1)
    rankUpdate!(-τ, x, v, A1)
    A
end

function lmul!(adjH::Adjoint{<:Any,<:Householder}, A::StridedMatrix)
    H = parent(adjH)
    m, n = size(A)
    size(H,1) == m || throw(DimensionMismatch("A: $m,$n H: $(size(H))"))
    v = view(H.v, 1:m - 1)
    τ = H.τ
    for j = 1:n
        va = A[1,j]
        Aj = view(A, 2:m, j)
        va += dot(v, Aj)
        va = τ'va
        A[1,j] -= va
        axpy!(-va, v, Aj)
    end
    A
end

Base.convert(::Type{Matrix}, H::Householder{T}) where {T} = lmul!(H, Matrix{T}(I, size(H, 1), size(H, 1)))
Base.convert(::Type{Matrix{T}}, H::Householder{T}) where {T} = lmul!(H, Matrix{T}(I, size(H, 1), size(H, 1)))
