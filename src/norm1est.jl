# This file is part of GenericSchur.jl, released under the MIT "Expat" license

# The method in this file is derived from LAPACK's zlacon.
# LAPACK is released under a BSD license, and is
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.


# Hager's one-norm estimator, with modifications by N.J. Higham
"""
    norm1est!(applyA!,applyAH!,y::Vector) => γ

Estimate the 1-norm of a linear operator `A` expressed as functions which
apply `A` and `adjoint(A)` to a vector such as `y`.

cf. N.J. Higham, SIAM J. Sci. Stat. Comp. 11, 804 (1990)
"""
function norm1est!(applyA!, applyAH!, y::AbstractVector{Ty}; maxiter=5) where Ty
    n = length(y)
    x = fill(one(Ty)/n,n)
    y .= zero(Ty)
    applyA!(x)
    (n == 1) && return abs(x[1])
    γ = norm(x,1)
    tiny = safemin(real(Ty))
    for i=1:n
        absxi = abs(x[i])
        if absxi > tiny
            x[i] /= absxi
        else
            x[i] = one(Ty)
        end
    end
    applyAH!(x)
    ax0,j0 = _findamax(x)
    for iter = 1:maxiter
        x .= zero(Ty)
        x[j0] = one(Ty)
        applyA!(x)
        y .= x
        oldγ = γ
        γ = norm(y,1)
        if γ <= oldγ
            break
        end
        for i=1:n
            absxi = abs(x[i])
            if absxi > tiny
                x[i] /= absxi
            else
                x[i] = one(Ty)
            end
        end
        applyAH!(x)
        jlast = j0
        ax0, j0 = _findamax(x)
        if abs(x[jlast]) == ax0
            break
        end
    end
    # alternative estimate for tricky cases (see Higham 1990)
    v = x # reuse workspace
    isign = 1
    for i in 1:n
        v[i] = isign * (1+(i-1)/(n-1))
        isign = -isign
    end
    applyA!(v)
    t = 2*norm(v,1) / (3*n)
    return max(t,γ)
end

function _findamax(x::AbstractVector{T}) where T
    ax0 = abs(x[1])
    i0 = 1
    for i=2:length(x)
        ax = abs(x[i])
        if ax > ax0
            ax0 = ax
            i0 = i
        end
    end
    return ax0,i0
end

function _findamax(x::AbstractVector{T}) where T <: Complex
    ax0 = abs2(x[1])
    i0 = 1
    for i=2:length(x)
        ax = abs2(x[i])
        if ax > ax0
            ax0 = ax
            i0 = i
        end
    end
    return sqrt(ax0),i0
end
