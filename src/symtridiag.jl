if VERSION <= v"1.12-"
    # name will be local
    struct RobustRepresentations <: LinearAlgebra.Algorithm end
else
    using LinearAlgebra: RobustRepresentations
end

const _HaveMRRR = Ref(false)
# allow another package to provide an implementation
function _registerMRRR()
    _HaveMRRR[] = true
end

function gschur!(H::Hessenberg{Tq, S},
                 Z::Union{AbstractMatrix, Nothing}=Matrix(H.Q),
                 alg=QRIteration();
                 kwargs...
                 ) where {S <: SymTridiagonal{T}} where {Tq <: Union{RT,Complex{RT}}, T <: Real} where {RT <: AbstractFloat}
    if isa(alg, RobustRepresentations) && !_HaveMRRR[]
        @warn """GenericSchur does not currently provide `alg=RobustRepresentations`
        (LinearAlgebra default); falling back to `QRIteration`.
        This adjustment will (probably) proceed without future warnings in this session.
        """ maxlog=1
        alg = QRIteration()
    end
    _gschur!(H.H, alg, Z; kwargs...)
    # aliasing might be a cruel trap, so don't.
    v = copy(H.H.dv)
    if Z === nothing
        return Schur(zeros(Tq,0,0),zeros(Tq,0,0),v)
    end
    # Schur form must have same type as Z
    n = length(v)
    Tschur = similar(Z,n,n)
    fill!(Tschur, zero(eltype(Z)))
    @inbounds for j=1:n
        Tschur[j,j] = v[j]
    end
    return Schur(Tschur, Z, v)
end

function geigen!(A::SymTridiagonal{T}, alg=QRIteration()) where T <: AbstractFloat
    n = length(A.dv)
    V = Matrix{T}(I,n,n)
    _gschur!(A, alg, V)
    λ = copy(A.dv)
    LinearAlgebra.Eigen(sorteig!(λ, V, eigsortby)...)
end

function geigvals!(A::SymTridiagonal{T}, alg=QRIteration()) where T <: AbstractFloat
    _gschur!(A, alg, nothing)
    λ = copy(A.dv)
    return λ
end

# the rest are internal implementation methods

function _gschur!(A::SymTridiagonal{T},
                  alg::Algorithm,
                  Z::Union{Nothing, AbstractArray} = nothing;
                  maxiter=30*size(A,1)) where {T}
    throw(ArgumentError("Unsupported algorithm $alg"))
end

#translated from LAPACK::zsteqr
# LAPACK Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
"""
     _gschur!(A::SymTridiagonal, ::QRIteration, Z)

    overwrites `A.dv` with eigenvalues, optionally applying orthogonal transforms to `Z`.
"""
function _gschur!(A::SymTridiagonal{T},
                  alg::QRIteration,
                  Z::Union{Nothing, AbstractArray} = nothing;
                  maxiter=30*size(A,1)) where {T}
    d = A.dv
    e = A.ev
    n = length(d)
    wantZ = (Z !== nothing) && (size(Z,1) > 0)
    if wantZ
        nz = size(Z,2)
        (nz == n) || throw(DimensionMismatch("second dimension of Z must match tridiagonal A"))
    end

    epsT = eps(T)
    eps2 = epsT^2
    safmin = floatmin(T)
    safmax = 1 / safmin
    ssfmax = sqrt(safmax) / 3
    ssfmin = sqrt(safmin) / eps2

    niter = 0
    gvec = Vector{Givens{T}}(undef, wantZ ? n : 0)

    # Determine where matrix splits and choose QL or QR for each block,
    # according to whether top or bottom diagonal element is smaller.

    l1 = 1
    while l1 <= n
        if l1 > 1
            e[l1-1] = zero(T)
        end
        msplit = _st_findsplit_fwd!(d,e,l1,n)
        l = l1
        lsave = l
        lend = msplit
        lendsave = lend
        l1 = msplit + 1
        if lend == l
            @mydebug println("empty at $l, continuing")
            continue
        end
        # scale A[l:lend,l:lend]
        if VERSION > v"1.12-"
            # FIXME: workaround for stdlib bug
            if lend == l + 1
                anorm = opnorm([d[l] e[l]; e[l] d[l+1]], Inf)
            else
                anorm = opnorm(SymTridiagonal(view(d, l:lend), view(e, l:lend-1)), Inf)
            end
        else
        anorm = opnorm(SymTridiagonal(view(d, l:lend), view(e, l:lend-1)), Inf)
        end
        iscale = 0
        if anorm == 0
            @mydebug println("null at $l:$lend, continuing")
            continue
        end
        if anorm > ssfmax
            iscale = 1
            safescale!(view(d, l:lend), anorm, ssfmax)
            safescale!(view(e, l:lend-1), anorm, ssfmax)
        elseif anorm < ssfmin
            iscale = 2
            safescale!(view(d, l:lend), anorm, ssfmin)
            safescale!(view(e, l:lend-1), anorm, ssfmin)
        end
        if abs(d[lend]) < abs(d[l])
            lend = lsave
            l = lendsave
        end

        if lend > l
            l, nit = _st_QLIter!(d,e,Z,l,lend,gvec)
        else
            l, nit = _st_QRIter!(d,e,Z,l,lend,gvec)
        end
        niter +- nit
        # undo scaling if necessary
        if iscale == 1
            safescale!(view(d, lsave:lendsave), ssfmax, anorm)
            safescale!(view(e, lsave:lendsave-1), ssfmax, anorm)
        elseif iscale == 2
            safescale!(view(d, lsave:lendsave), ssfmin, anorm)
            safescale!(view(e, lsave:lendsave-1), ssfmin, anorm)
        end

        # check for iteration limit
        if niter >= maxiter
            nbad = count(x -> !iszero(x), e)
            @warn "$nbad eigenvalues failed to converge in $maxiter iterations"
            return
        end
        @mydebug begin
            if l1 <= n
                println("iscale was $iscale; next: l1=$l1")
            else
                println("iscale was $iscale; done.")
            end
        end
    end # outer loop
    nothing
end

# QL iteration
function _st_QLIter!(d::AbstractVector{T},e,Z,l,lend,gvec) where {T}
    @mydebug println("QL $l:$lend")
    wantZ = Z !== nothing
    safmin = floatmin(T)
    eps2 = eps(T)^2
    niter = 0
    while l <= lend
        # find small subdiagonal
        msplit = lend
        for m=l:lend-1
            if abs2(e[m]) <= (eps2 * abs(d[m])) * abs(d[m+1]) + safmin
                msplit = m
                break
            end
        end
        if msplit < lend
            e[msplit] = zero(T)
        end
        p = d[l]
        if msplit != l
            # If remaining matrix is 2x2, diagonalize it
            if msplit == l+1
                @mydebug println("deflating 2 at $l")
                if wantZ
                    rt1, rt2, cs, sn = _dgz_sym2x2(d[l],e[l],d[l+1], true)
                    G = Givens(l,l+1,cs,sn)
                    gvec[l] = G
                    rmul!(Z, G')
                else
                    rt1, rt2 = _dgz_sym2x2(d[l],e[l],d[l+1], false)
                end
                d[l] = rt1
                d[l+1] = rt2
                e[l] = zero(T)
                l += 2
                if l <= lend
                    continue # to top of QL loop
                end
                break
            end
            niter += 1
            # compute shift
            g = (d[l+1] - p) / (2 * e[l])
            r = hypot(g, one(T))
            g = d[msplit] - p + (e[l] / (g + copysign(r,g)))
            s = one(T)
            c = one(T)
            p = zero(T)
            # inner loop
            for i = msplit-1:-1:l
                f = s * e[i]
                b = c * e[i]
                c,s,r = givensAlgorithm(g,f)
                if i < msplit-1
                    e[i+1] = r
                end
                g = d[i+1] - p
                r = (d[i] - g) * s + 2 * c * b
                p = s * r
                d[i+1] = g+p
                g = c*r - b
                if wantZ
                    gvec[i] = Givens(i,i+1,c, -s)
                end
            end
            if wantZ
                # xlasr RVB
                for i=msplit-1:-1:l
                    rmul!(Z, gvec[i]')
                end
            end
            d[l] -= p
            e[l] = g
        else
            d[l] = p
            l += 1
        end
    end # l loop for QL
    @mydebug println("QL end")
    return l, niter
end

# QR iteration
function _st_QRIter!(d::AbstractVector{T},e,Z,l,lend,gvec) where {T}
    @mydebug println("QR $lend:$l")
    wantZ = Z !== nothing
    safmin = floatmin(T)
    eps2 = eps(T)^2
    niter = 0
    while l >= lend
        msplit = lend
        for m = l:-1:lend+1
            if abs2(e[m-1]) <= (eps2 * abs(d[m])) * abs(d[m-1]) + safmin
                msplit = m
                break
            end
        end
        if msplit > lend
            e[msplit-1] = zero(T)
        end
        p = d[l]
        if msplit != l
            # If remaining matrix is 2x2, diagonalize it
            if msplit == l-1
                @mydebug println("deflating 2 at $msplit")
                if wantZ
                    rt1, rt2, cs,sn = _dgz_sym2x2(d[l-1],e[l-1],d[l], true)
                    G = Givens(l-1,l,cs,sn)
                    gvec[msplit] = G
                    # xlasr RVF
                    rmul!(Z, G')
                else
                    rt1, rt2, _, _ = _dgz_sym2x2(d[l-1],e[l-1],d[l], false)
                end
                d[l-1] = rt1
                d[l] = rt2
                e[l-1] = zero(T)
                l -= 2
                if l >= lend
                    continue # to top of QR loop
                end
                break
            end
            niter += 1
            # compute shift
            g = (d[l-1] - p) / (2 * e[l-1])
            r = hypot(g, one(T))
            g = d[msplit] - p + (e[l-1] / (g + copysign(r,g)))
            s = one(T)
            c = one(T)
            p = zero(T)
            # inner loop
            for i = msplit:l-1
                f = s * e[i]
                b = c * e[i]
                c,s,r = givensAlgorithm(g,f)
                if i != msplit
                    e[i-1] = r
                end
                g = d[i] - p
                r = (d[i+1] - g) * s + 2 * c * b
                p = s * r
                d[i] = g+p
                g = c*r - b
                if wantZ
                    gvec[i] = Givens(i,i+1,c, s)
                end
            end
            if wantZ
                # xlasr RVF
                for i=msplit:l-1
                    rmul!(Z, gvec[i]')
                end
            end
            d[l] -= p
            e[l-1] = g
        else
            d[l] = p
            l -= 1
        end
    end # l loop for QR
    @mydebug println("QR end")
    return l, niter
end

@inline function _st_findsplit_fwd!(d::AbstractVector{T},e,l1,n) where T
    msplit = n
    if l1 <= n-1
        for m=l1:n-1
            tst = abs(e[m])
            if iszero(tst)
                msplit = m
                break
            end
            if tst <= sqrt(abs(d[m])) * sqrt(abs(d[m+1])) * eps(T)
                e[m] = zero(T)
                msplit = m
                break
            end
        end
    end
    return msplit
end


function _dgz_sym2x2(a::T, b, c, wantvec) where T
    sm = a+c
    df = a-c
    adf = abs(df)
    tb = 2b
    ab = abs(tb)
    (acmx, acmn) = (abs(a) > abs(c)) ? (a,c) : (c,a)
    rt = hypot(ab,adf)
    if sm < zero(T)
        rt1 = (sm - rt) / 2
        sgn1 = -1
        # order of execution important
        # for full accuracy, next line needs higher precision
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b
    elseif sm > zero(T)
        rt1 = (sm + rt) / 2
        sgn1 = 1
        # order of execution important
        # for full accuracy, next line needs higher precision
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b
    else
        rt1, rt2 = rt/2, -rt/2
        sgn1 = 1
    end
    if wantvec
        # compute eigenvector
        if df > 0
            cs = df + rt
            sgn2 = 1
        else
            cs = df - rt
            sgn2 = -1
        end
        acs = abs(cs)
        if acs > ab
            ct = -tb / cs
            sn1 = one(T) / sqrt(one(T) + ct^2)
            cs1 = ct * sn1
        else
            if iszero(ab)
                cs1 = one(T)
                sn1 = zero(T)
            else
                tn = -cs / tb
                cs1 = one(T) / sqrt(one(T) + tn^2)
                sn1 = tn * cs1
            end
        end
        if sgn1 == sgn2
            tn = cs1
            cs1 = -sn1
            sn1 = tn
        end
        return rt1, rt2, cs1, sn1
    else
        return rt1, rt2, zero(T), zero(T)
    end
end
