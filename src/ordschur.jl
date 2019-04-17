# This file is part of GenericSchur.jl, released under the MIT "Expat" license
# Portions derived from LAPACK, see below.

# invariant subspace computations using complex Schur decompositions

import LinearAlgebra: _ordschur!, _ordschur

function _ordschur(T::StridedMatrix{Ty},
                    Z::StridedMatrix{Ty},
                   select::Union{Vector{Bool},BitVector}) where Ty <: Complex
    _ordschur!(copy(T), copy(Z), select)
end

function _ordschur!(T::StridedMatrix{Ty},
                    Z::StridedMatrix{Ty},
                    select::Union{Vector{Bool},BitVector}) where Ty <: Complex
    # suppress most checks since this is an internal function expecting
    # components of a Schur object
    n = size(T,1)
    ks = 0
    for k=1:n
        if select[k]
            ks += 1
            if k != ks
                _trexchange!(T,Z,k,ks)
            end
        end
    end
    triu!(T)
    T,Z,diag(T)
end

"""
reorder `T` by a unitary similarity transformation so that `T[iold,iold]`
is moved to `T[inew,inew]`. Also right-multiply `Z` by the same transformation.
"""
function _trexchange!(T,Z,iold,inew)
    # this is Algorithm 7.6.1 in Golub & van Loan 4th ed.
    n = size(T,1)
    if (n < 1) || (iold == inew)
        return
    end
    if iold < inew
        krange = iold:inew-1
    else
        krange = iold-1:-1:inew
    end
    for k in krange
        Tkk = T[k,k]
        Tpp = T[k+1,k+1]
        G,_ = givens(T[k,k+1], Tpp - Tkk, k, k+1)
        if k+1 <= n
            lmul!(G,T)
        end
        rmul!(T,adjoint(G))
        Z === nothing || rmul!(Z,adjoint(G))
    end
end

# eigvalscond and subspacesep are derived from LAPACK's ztrsen.
# LAPACK is released under a BSD license, and is
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.

"""
    eigvalscond(S::Schur,nsub::Integer) => Real

Estimate the reciprocal of the condition number of the `nsub` leading
eigenvalues of `S`. (Use `ordschur` to move a subspace of interest
to the front of `S`.)

See the LAPACK User's Guide for details of interpretation.
"""
function eigvalscond(S::Schur{Ty},nsub) where Ty
    n = size(S.T,1)
    # solve T₁₁ R - R T₂₂ = σ T₁₂
    R = S.T[1:nsub,nsub+1:n] # copy
    R, scale = trsylvester!(view(S.T,1:nsub,1:nsub),view(S.T,nsub+1:n,nsub+1:n),
                    R)
    rnorm = norm(R) # Frobenius norm, as desired
    s = (rnorm > 0) ?
        scale / (sqrt(scale*scale / rnorm + rnorm ) * sqrt(rnorm)) :
        one(real(Ty))
    s
end

"""
    subspacesep(S::Schur,nsub::Integer) => Real

Estimate the reciprocal condition of the separation angle for the
invariant subspace corresponding
to the leading block of size `nsub` of a Schur decomposition.
(Use `ordschur` to move a subspace of interest
to the front of `S`.)

See the LAPACK User's Guide for details of interpretation.
"""
function subspacesep(S::Schur{Ty}, nsub) where Ty
    n = size(S.T,1)
    scale = one(real(Ty))
    function f(X)
        # solve T₁₁ R - R T₂₂ = σ X
        R, s = trsylvester!(view(S.T,1:nsub,1:nsub),
                            view(S.T,nsub+1:n,nsub+1:n),
                            reshape(X,nsub,n-nsub))
        scale = s
        R
    end
    function fH(X)
        # solve T₁₁ᴴ R - R T₂₂ᴴ = σ X
        R, s = adjtrsylvester!(view(S.T,1:nsub,1:nsub),
                               view(S.T,nsub+1:n,nsub+1:n),
                               reshape(X,nsub,n-nsub))
        scale = s
        R
    end
    est = norm1est!(f,fH,zeros(Ty,nsub*(n-nsub)))
    return scale / est
end
