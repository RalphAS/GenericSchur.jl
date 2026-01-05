# This file is part of GenericSchur.jl, released under the MIT "Expat" license

# The methods in this file are derived from LAPACK's ztrsyl.
# LAPACK is released under a BSD license, and is
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.

function trsylvester!(
        A::UpperTriangular{T, S}, B::UpperTriangular{T, S},
        C::StridedVecOrMat{T};
        possign = false
    ) where {T, S <: StridedMatrix{T}}
    return trsylvester!(A.data, B.data, C, possign = possign)
end

"""
    trsylvester!(A,B,C) => X, σ

solve the Sylvester equation ``A X - X B = σ C``
for upper triangular and square `A` and `B`, overwriting `C`
and setting `σ` to avoid overflow.
"""
function trsylvester!(A::StridedMatrix{T}, B::StridedMatrix{T}, C::StridedVecOrMat{T}; possign = false) where {T}
    m = checksquare(A)
    n = checksquare(B)
    ((size(C, 1) == m) && (size(C, 2) == n)) || throw(
        DimensionMismatch(
            "dimensions of C $(size(C)) must match A, ($m,$m), and B, ($n,$n)"
        )
    )

    scale = one(real(T))

    tiny = eps(real(T))
    small = safemin(real(T)) * m * n / tiny
    bignum = one(real(T)) / small
    smin = max(small, tiny * norm(A, Inf) * norm(B, Inf))

    isgn = possign ? one(real(T)) : -one(real(T))
    ierr = 0
    for l in 1:n
        for k in m:-1:1
            # FIXME: these should be dotu(), but I can't find anything usable in stdlib::LA
            # WARNING: this relies on isempty() pass in checkindex() for view()
            suml = sum(view(A, k, (k + 1):m) .* view(C, (k + 1):m, l))
            sumr = sum(view(C, k, 1:(l - 1)) .* view(B, 1:(l - 1), l))
            v = C[k, l] - (suml + isgn * sumr)
            scaloc = one(real(T))
            a11 = A[k, k] + isgn * B[l, l]
            da11 = abs1(a11)
            if da11 <= smin
                a11 = smin
                da11 = smin
                ier = 1
            end
            db = abs1(v)
            if (da11 < 1) && (db > 1)
                if (db > bignum * da11)
                    # println("scaling by $db")
                    scaloc = 1 / db
                end
            end
            x11 = (v * scaloc) / a11
            if scaloc != 1
                lmul!(scaloc, C)
                scale *= scaloc
            end
            C[k, l] = x11
        end
    end
    return C, scale
end

function adjtrsylvester!(
        A::UpperTriangular{T, S}, B::UpperTriangular{T, S},
        C::StridedVecOrMat{T};
        possign = false
    ) where {T, S <: StridedMatrix{T}}
    return adjtrsylvester!(A.data, B.data, C, possign = possign)
end

"""
    adjtrsylvester!(A,B,C) => X, σ

solve the Sylvester equation ``Aᴴ X - X Bᴴ = σ C``,
for upper triangular `A` and `B`, overwriting `C`
and setting `σ` to avoid overflow.
"""
function adjtrsylvester!(A::StridedMatrix{T}, B::StridedMatrix{T}, C::StridedVecOrMat{T}; possign = false) where {T}
    m = checksquare(A)
    n = checksquare(B)
    ((size(C, 1) == m) && (size(C, 2) == n)) || throw(
        DimensionMismatch(
            "dimensions of C $(size(C)) must match A, ($m,$m), and B, ($n,$n)"
        )
    )

    scale = one(real(T))

    tiny = eps(real(T))
    small = safemin(real(T)) * m * n / tiny
    bignum = one(real(T)) / small
    smin = max(small, tiny * norm(A, Inf) * norm(B, Inf))

    isgn = possign ? one(real(T)) : -one(real(T))
    for l in n:-1:1
        for k in 1:m
            suml = dot(view(A, 1:(k - 1), k), view(C, 1:(k - 1), l))
            # WARNING: this relies on isempty() pass in checkindex() for view()
            sumr = dot(view(C, k, (l + 1):n), view(B, l, (l + 1):n))
            v = C[k, l] - (suml + isgn * conj(sumr))
            scaloc = one(real(T))
            a11 = conj(A[k, k] + isgn * B[l, l])
            da11 = abs1(a11)
            if da11 < smin
                a11 = smin
                da11 = smin
                ier = 1
            end
            db = abs1(v)
            if (da11 < 1) && (db > 1)
                if (db > bignum * da11)
                    scaloc = 1 / db
                end
            end
            x11 = (v * scaloc) / a11
            if scaloc != 1
                lmul!(scaloc, C)
                scale *= scaloc
            end
            C[k, l] = x11
        end
    end
    return C, scale
end
