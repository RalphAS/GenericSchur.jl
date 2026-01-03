# This file is part of GenericSchur.jl, released under the MIT "Expat" license
# Portions derived from LAPACK, see below.

function gordschur!(F::Schur, select::Union{Vector{Bool},BitVector})
    T,Z,λ = gordschur!(F.T, F.Z, select)
    Schur(T,Z,λ)
end
function gordschur!(F::GeneralizedSchur, select::Union{Vector{Bool},BitVector})
    S,T,α,β,Q,Z = gordschur!(F.S, F.T, F.Q, F.Z, select)
    GeneralizedSchur(S,T,α,β,Q,Z)
end

# invariant subspace computations using complex Schur decompositions

function gordschur!(T::StridedMatrix{Ty},
                    Z::StridedMatrix{Ty},
                    select::Union{Vector{Bool},BitVector}
) where {Ty <: Complex}
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
`_trexchange!(T,Z,iold,inew)`

Given an upper triangular T and unitary Z, reorder `T` by a unitary similarity
transformation so that `T[iold,iold]`
is moved to `T[inew,inew]`. Also right-multiply `Z` by the same transformation.
"""
function _trexchange!(T::AbstractMatrix{Ty}, Z, iold, inew) where {Ty <: Complex}
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
        isempty(Z) || rmul!(Z,adjoint(G))
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

# Reordering generalized Schur decompositions, Complex element type

function gordschur!(S::StridedMatrix{Ty}, T::StridedMatrix{Ty},
                    Q::StridedMatrix{Ty}, Z::StridedMatrix{Ty},
                    select::Union{Vector{Bool},BitVector};
                    scale::Bool=true
                    ) where {Ty <: Complex}
    n = size(S,1)
    if scale
        scaleA, cscale, anrm = _scale!(S)
        scaleB, cscaleb, bnrm = _scale!(T)
    else
        scaleA = false
        scaleB = false
    end
    ks = 0
    for k=1:n
        if select[k]
            ks += 1
            if k != ks
                _trexchange!(S,T,Q,Z,k,ks)
            end
        end
    end
    if scaleA
        safescale!(S, cscale, anrm)
    end
    if scaleB
        safescale!(T, cscaleb, bnrm)
    end
    triu!(S)
    triu!(T)
    S,T,diag(S),diag(T),Q,Z
end

"""
`_trexchange!(A,B,Q,Z,iold,inew)`

Move location `iold` to `inew` in a generalized Schur decomposition by unitary
transformations.
"""
function _trexchange!(A::AbstractMatrix{Ty}, B, Q, Z, iold, inew) where {Ty <: Complex}
    if iold < inew
        icur = iold
        while icur < inew
            _trexch2!(A,B,Q,Z,icur)
            icur += 1
        end
        icur -= 1
    else
        @mydebug println("reverse swap $iold with $inew")
        icur =  iold-1
        while icur >= inew
            _trexch2!(A,B,Q,Z,icur)
            icur -= 1
        end
        icur += 1
    end
    nothing
end

"""
`_trexch2!(A,B,Q,Z,j1)`

swap adjacent 1x1 blocks in UT pair (A,B) via unitary equivalence transform
apply same transform to associated Q,Z
"""
function _trexch2!(A::AbstractMatrix{Ty},B,Q,Z,j1;
                   require_strong=true, throws=true
) where {Ty <: STypes}
    # based on ztgex2
    # copy the block to try the swap
    S = A[j1:j1+1,j1:j1+1]
    T = B[j1:j1+1,j1:j1+1]
    # compute threshold for acceptance
    Tr = real(Ty)
    smlnum = safemin(Tr) / eps(Tr)
    W = hcat(S,T)
    sa = _safe_fnorm(vec(W))
    thresh = max(Tr(20)*eps(Tr)*sa, smlnum)

    # compute Givens rotations which would swap blocks
    # and tentatively do it
    f = S[2,2] * T[1,1] - T[2,2] * S[1,1]
    g = S[2,2] * T[1,2] - T[2,2] * S[1,2]
    sa = abs(S[2,2]) * abs(T[1,1])
    sb = abs(S[1,1]) * abs(T[2,2])
    if f == 0 && g == 0
        cz, sz = zero(Ty), one(Ty)
    else
        cz, sz, _ = givensAlgorithm(g,f)
    end
    Gz1 = Givens(1,2,Ty(cz),-sz)
    rmul!(S,adjoint(Gz1))
    rmul!(T,adjoint(Gz1))
    if sa >= sb
        fq, gq = S[1,1], S[2,1]
    else
        fq, gq = T[1,1], T[2,1]
    end
    if sz != 0 && fq == 0 && gq == 0
        # patch for continuity (edge case, not handled thus in LAPACK)
        # CHECKME: the condition may not be optimal
        cq, sq = zero(Ty), one(Ty)
    else
        cq, sq, _ = givensAlgorithm(fq, gq)
    end
    Gq1 = Givens(1,2,Ty(cq),sq)
    lmul!(Gq1,S)
    lmul!(Gq1,T)
    # weak stability test: subdiags <= O( ϵ norm((S,T),"F"))
    ws = abs(S[2,1]) + abs(T[2,1])
    @mydebug println("  weak test: subdiag norm = $ws")
    weak_ok = ws <= thresh

    if !weak_ok
        @mydebug println("  failed weak stability test")
        throws && throw(IllConditionException(j1))
        return false
    end

    if require_strong
        # Strong stability test
        # Supposedly equiv. to
        # |[A-Qᴴ*S*Z, B-Qᴴ*T*Z]| <= O(1) ϵ |[A,B]| in F-norm
        Ss = copy(S)
        Ts = copy(T)
        Gz2 = Givens(1,2,Ty(cz),sz)
        rmul!(Ss,adjoint(Gz2))
        rmul!(Ts,adjoint(Gz2))
        Gq2 = Givens(1,2,Ty(cq),-sq)
        lmul!(Gq2,Ss)
        lmul!(Gq2,Ts)
        Ss .-= A[j1:j1+1,j1:j1+1]
        Ts .-= B[j1:j1+1,j1:j1+1]
        W = hcat(Ss,Ts)
        ss = _safe_fnorm(vec(W))
        @mydebug println("  strong test resid=$ss [thresh=$thresh]")
        strong_ok = ss <= thresh
        if !strong_ok
            @mydebug println("  failed strong stability test")
            throws && throw(IllConditionException(j1))
            return false
        end
    end

    Gz = Givens(j1,j1+1,Ty(cz),-sz)
    rmul!(A,adjoint(Gz))
    rmul!(B,adjoint(Gz))
    Gq = Givens(j1,j1+1,Ty(cq),sq)
    lmul!(Gq,A)
    lmul!(Gq,B)
    A[j1+1,j1] = zero(Ty)
    B[j1+1,j1] = zero(Ty)
    rmul!(Z,adjoint(Gz))
    rmul!(Q,adjoint(Gq))
    return true
end
