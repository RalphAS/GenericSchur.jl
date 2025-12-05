# This file is part of GenericSchur.jl, released under the MIT "Expat" license

# The methods in this file are derived from LAPACK's ztgsyl etc.
# LAPACK is released under a BSD license, and is
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.

# Note: since this is based on the routines for complex types,
# several methods are hard-coded with 2x2 blocks

"""
    trsylvester!(A,B,C,D,E,F) => R, L, σ

solve the generalized Sylvester equation ``A R - L B = σ C``
``D R - L E = σ F``
given upper triangular and square `A,B,D` and `E`.
Overwrites `C` and `F`
with `R` and `L`, and sets `σ` to avoid overflow.
"""
function trsylvester!(A::StridedMatrix{T},B::StridedMatrix{T},
                      C::StridedVecOrMat{T},
                      D::StridedMatrix{T},E::StridedMatrix{T},
                      F::StridedVecOrMat{T}) where {T}
    m = checksquare(A)
    n = checksquare(B)
    ((size(C,1) == m) && (size(C,2) == n)) || throw(DimensionMismatch(
        "dimensions of C $(size(C)) must match A, ($m,$m), and B, ($n,$n)"))
    m1 = checksquare(D)
    n1 = checksquare(E)
    (m1 == m && n1 == n) || throw(DimensionMismatch(
        "dimensions of D and E must match A and B"))
    ((size(F,1) == m) && (size(F,2) == n)) || throw(DimensionMismatch(
        "dimensions of F $(size(F)) must match A, ($m,$m), and B, ($n,$n)"))

    rtyone = one(real(T))
    rtyzero = zero(real(T))
    scale = rtyone
    scaloc = rtyone
    dif = rtyzero
    cone = one(T)
    rdsum = rtyone
    rdscale = rtyzero
    # ztgsyl notes:
    # ijob 0: solve only; 1: solve & look-ahead dif; 2: solve & 1-est dif
    #      3: only look-ahead dif; 4: only 1-est dif
    # ifunc 1: look-ahead, 2: 1-est; else 0 (reset for 2-pass cases)
    #      ifunc => ijob in ztgsy2
    # isolve 2: solve and (some) dif; else 1
    # if solving:
    #   solve for C,F
    # if dif:
    #   if solving, save C,F
    #   zero out C,F
    #   call solver w/ ifunc
    #   if solving, restore C,F
    for j=1:n
        for i=m:-1:1
            # build 2x2 problem
            Z = [A[i,i] -B[j,j]; D[i,i] -E[j,j]]
            rhs = [C[i,j]; F[i,j]]
            x, unperturbed, scl = _safe_lu_solve!(Z, rhs)
            C[i,j] = x[1]
            F[i,j] = x[2]
            if scl != 1
                lmul!(view(C, 1:m, k), scl)
                lmul!(view(F, 1:m, k), scl)
            end
            # substitute R[i,j], L[i,j] into remaining eq
            if i > 1
                α = -x[1]
                C[1:i-1,j] .+= α * A[1:i-1,i]
                F[1:i-1,j] .+= α * D[1:i-1,i]
            end
            if j < n
                α = x[2]
                C[i,j+1:n] .+= α * B[j,j+1:n]
                F[i,j+1:n] .+= α * E[j,j+1:n]
            end
        end
    end
    return C, F, scale
end

"""
    adjtrsylvester!(A,B,C,D,E,F) => X, σ

solve a generalized adjoint Sylvester equation ``Aᴴ X + Dᴴ Y = σ C``,
 ``X Bᴴ + Y Eᴴ = -σ F``,
for upper triangular `A, B, D,` and `E`, overwriting `C` and `F`
and setting `σ` to avoid overflow.
"""
function adjtrsylvester!(A::StridedMatrix{T},B::StridedMatrix{T},
                      C::StridedVecOrMat{T},
                      D::StridedMatrix{T},E::StridedMatrix{T},
                      F::StridedVecOrMat{T}) where {T}
    m = checksquare(A)
    n = checksquare(B)
    ((size(C,1) == m) && (size(C,2) == n)) || throw(DimensionMismatch(
        "dimensions of C $(size(C)) must match A, ($m,$m), and B, ($n,$n)"))
    m1 = checksquare(D)
    n1 = checksquare(E)
    (m1 == m && n1 == n) || throw(DimensionMismatch(
        "dimensions of D and E must match A and B"))
    ((size(F,1) == m) && (size(F,2) == n)) || throw(DimensionMismatch(
        "dimensions of F $(size(F)) must match A, ($m,$m), and B, ($n,$n)"))

    scale = one(real(T))
    scaloc = one(real(T))
    dif = zero(real(T))
    cone = one(T)
    for i=1:m
        for j=n:-1:1
            # build 2x2 problem
            Z = [A[i,i]' D[i,i]'; -B[j,j]' -E[j,j]']
            rhs = [C[i,j]; F[i,j]]
            # FIXME: use safe version and update scale
            fZ = lu(Z)
            x = fZ \ rhs
            C[i,j] = x[1]
            F[i,j] = x[2]
            # substitute R[i,j], L[i,j] into remaining eq
            for jj=1:j-1
                F[i,jj] += x[1] * B[jj,j]' + x[2] * E[jj,j]'
            end
            for ii=i+1:m
                C[ii,j] -= x[1] * A[i,ii]' + x[2] * D[i,ii]'
            end
        end
    end
    return C, F, scale
end
