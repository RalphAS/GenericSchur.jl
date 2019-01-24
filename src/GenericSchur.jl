module GenericSchur
using LinearAlgebra
using LinearAlgebra: Givens, Rotation
using Printf
import LinearAlgebra: lmul!, mul!, checksquare

# This is the public interface of the package.
# Wrappers like `schur` and `eigvals` should just work.
import LinearAlgebra: schur!, eigvals!, eigvecs, eigen!
export triangularize, eigvalscond, subspacesep

schur!(A::StridedMatrix{T}; kwargs...) where {T} = gschur!(A; kwargs...)

function eigvals!(A::StridedMatrix{T}; kwargs...) where {T}
    S = gschur!(A; wantZ=false, kwargs...)
    S.values
end

# This is probably the best we can do unless LinearAlgebra coöperates
"""
    eigvecs(S::Schur{<:Complex}; left=false)

Compute right or left eigenvectors from a Schur decomposition.
"""
function eigvecs(S::Schur{T}; left::Bool=false) where {T <: Complex}
    if left
        v = _gleigvecs!(S.T,S.Z)
    else
        v = _geigvecs!(S.T,S.Z)
    end
    v
end

function eigen!(A::StridedMatrix{T}; permute=true, scale=true) where {T <: Real}
    if permute
        throw(ArgumentError("permute=true is not available for generic Schur"))
    end
    if !scale
        @warn "scaling is always on for generic Schur"
    end
    S = triangularize(schur(A))
    v = eigvecs(S)
    LinearAlgebra.Eigen(S.values,v)
end

function eigen!(A::StridedMatrix{T}; permute=true, scale=true) where {T <: Complex}
    if permute
        throw(ArgumentError("permute=true is not available for generic Schur"))
    end
    if !scale
        @warn "scaling is always on for generic Schur"
    end
    S = schur(A)
    v = eigvecs(S)
    LinearAlgebra.Eigen(S.values,v)
end

############################################################################
# Internal implementation follows


include("util.jl")
include("hessenberg.jl")
include("householder.jl")

function _gschur!(H::HessenbergFactorization{T}, Z=nothing;
                 debug = false,
                 maxiter = 100*size(H, 1), maxinner = 30*size(H, 1), kwargs...
                 ) where {T <: Complex}
    n = size(H, 1)
    istart = 1
    iend = n
    w = Vector{T}(undef, n)
    HH = H.data

    RT = real(T)
    ulp = eps(RT)
    smallnum = safemin(RT) * (n / ulp)
    rzero = zero(RT)
    half = 1 / RT(2)
    threeq = 3 / RT(4)

    # iteration count
    it = 0

    @inbounds while iend >= 1
        istart = 1
        for its=0:maxinner
            it += 1
            if it > maxiter
                throw(ArgumentError("iteration limit $maxiter reached"))
            end

            # Determine if the matrix splits.
            # Find lowest positioned subdiagonal "zero" if any; reset istart
            for _istart in iend - 1:-1:istart
                debug && @printf("Subdiagonal element is: %10.3e%+10.3eim and istart,iend now %6d:%6d\n", reim(HH[_istart+1, _istart])..., istart,iend)
                if abs1(HH[_istart + 1, _istart]) <= smallnum
                    istart = _istart + 1
                    debug && @printf("Split1! Subdiagonal element is: %10.3e%+10.3eim and istart now %6d\n", reim(HH[istart, istart - 1])..., istart)
                    break
                end
                # deflation criterion from Ahues & Tisseur (LAWN 122, 1997)
                tst = abs1(HH[_istart,_istart]) + abs1(HH[_istart+1,_istart+1])
                if tst == 0
                    if (_istart-1 >= 1) tst += abs(real(HH[_istart,_istart-1])) end
                    if (_istart+2 <= n) tst += abs(real(HH[_istart+2,_istart+1])) end
                end
                if abs(real(HH[_istart+1,_istart])) <= ulp*tst
                    ab = max(abs1(HH[_istart+1,_istart]),
                             abs1(HH[_istart,_istart+1]))
                    ba = min(abs1(HH[_istart+1,_istart]),
                             abs1(HH[_istart,_istart+1]))
                    aa = max(abs1(HH[_istart+1,_istart+1]),
                             abs1(HH[_istart,_istart]-HH[_istart+1,_istart+1]))
                    bb = min(abs1(HH[_istart+1,_istart+1]),
                             abs1(HH[_istart,_istart]-HH[_istart+1,_istart+1]))
                    s = aa + ab
                    if ba * (ab / s) <= max(smallnum, ulp * (bb * (aa / s)))
                        istart = _istart + 1
                        debug && @printf("Split2! Subdiagonal element is: %10.3e%+10.3eim and istart now %6d\n", reim(HH[istart, istart - 1])..., istart)
                        break
                    end
                end
                # istart = 1
            end # check for split

            if istart > 1
                # clean up
                HH[istart, istart-1] = zero(T)
            end

            # if block size is one we deflate
            if istart >= iend
                debug && @printf("Bottom deflation! Block size is one. New iend is %6d\n", iend - 1)
                iend -= 1
                break
            end

            # select shift
            # logic adapted from LAPACK zlahqr
            if its % 30 == 10
                s = threeq * abs(real(HH[istart+1,istart]))
                t = s + HH[istart,istart]
            elseif its % 30 == 20
                s = threeq * abs(real(HH[iend,iend-1]))
                t = s + HH[iend,iend]
            else
                t = HH[iend,iend]
                u = sqrt(HH[iend-1,iend]) * sqrt(HH[iend,iend-1])
                s = abs1(u)
                if s ≠ rzero
                    x = half * (HH[iend-1,iend-1] - t)
                    sx = abs1(x)
                    s = max(s, abs1(x))
                    y = s * sqrt( (x/s)^2 + (u/s)^2)
                    if sx > rzero
                        if real(x / sx) * real(y) + imag(x / sx) * imag(y) < rzero
                            y = -y
                        end
                    end
                    t -= u * (u / (x+y))
                end
            end # shift selection

            # run a QR iteration
            debug && @printf("block start is: %6d, block end is: %6d, t: %10.3e%+10.3eim\n", istart, iend, reim(t)...)
            # zlahqr only has single-shift
            singleShiftQR!(HH, Z, t, istart, iend)

        end # inner loop
    end # outer loop
    w = diag(HH)
    return Schur(triu(HH), Z === nothing ? Matrix{T}(undef,0,0) : Z, w)
end

"""
gschur(A::StridedMatrix) -> F::Schur

Computes the Schur factorization of matrix `A` using a generic implementation.
See `LinearAlgebra.schur` for usage.
"""
gschur(A::StridedMatrix{T}; kwargs...) where {T} = gschur!(Matrix(A); kwargs...)

"""
gschur!(A::StridedMatrix) -> F::Schur

Destructive version of `gschur` (q.v.).
"""
function gschur!(A::StridedMatrix{T}; wantZ::Bool=true, scale::Bool=true,
                 permute::Bool=false, kwargs...) where T <: Complex
    n = checksquare(A)
    # FIXME: some LinearAlgebra wrappers force default permute=true
    # so we must silently ignore it here.
#    permute &&
#        throw(ArgumentError("permute option is not available for this method"))
    if scale
        scaleA, cscale, anrm = _scale!(A)
    else
        scaleA = false
    end
    H = _hessenberg!(A)
    if wantZ
        τ = H.τ # Householder reflectors w/ scales
        Z = Matrix{T}(I, n, n)
        for j=n-1:-1:1
            lmul!(τ[j], view(Z, j+1:n, j:n))
            Z[1:j-1,j] .= 0
        end
        S = _gschur!(H, Z; kwargs...)
    else
        S = _gschur!(H; kwargs...)
    end
    if scaleA
        safescale!(S.T, cscale, anrm)
        S.values .= diag(S.T, 0)
    end
    S
end

# Note: zlahqr exploits the fact that some terms are real to reduce
# arithmetic load.  Does that also work with Givens version?
# Is it worth the trouble?

function singleShiftQR!(HH::StridedMatrix{T}, Z, shift::Number, istart::Integer, iend::Integer) where {T <: Complex}
    m = size(HH, 1)
    ulp = eps(real(eltype(HH)))

    # look for two consecutive small subdiagonals
    istart1 = -1
    h11s = zero(eltype(HH))
    h21 = zero(real(eltype(HH)))
    for mm = iend-1:-1:istart+1
        # determine the effect of starting the single-shift Francis
        # iteration at row mm: see if this would make HH[mm,mm-1] tiny.
        h11 = HH[mm,mm]
        h22 = HH[mm+1,mm+1]
        h11s = h11 - shift
#        h21 = real(HH[mm+1,mm]) # for reflector
        h21 = HH[mm+1,mm]
        s = abs1(h11s) + abs(h21)
        h11s /= s
        h21 /= s
        h10 = real(HH[mm,mm-1])
        if abs(h10)*abs(h21) <= ulp *
            (abs1(h11s)*(abs1(h11)+abs1(h22)))
            istart1 = mm
            break
        end
    end
    if istart1 < 1
        istart1 = istart
        h11 = HH[istart,istart]
        h22 = HH[istart+1,istart+1]
        h11s = h11 - shift
        # h21 = real(HH[istart+1,istart]) # for reflector
        h21 = HH[istart+1,istart]
        s = abs1(h11s) + abs(h21)
        h11s /= s
        h21 /= s
    end

    if m > istart1 + 1
        Htmp = HH[istart1 + 2, istart1]
        HH[istart1 + 2, istart1] = 0
    end

    # create a bulge
    G, _ = givens(h11s, h21, istart1, istart1 + 1)
    lmul!(G, view(HH, :, istart1:m))
    rmul!(view(HH, 1:min(istart1 + 2, iend), :), G')
    Z === nothing || rmul!(Z, G')
    # do we need this? LAPACK uses Householder so some work would be needed
    # if istart1 > istart
        # if two consecutive small subdiagonals were found, scale
        # so HH[istart1,istart1-1] remains real.
    # end

    # chase the bulge down
    for i = istart1:iend - 2
        # i is K-1, istart is M
        G, _ = givens(HH[i + 1, i], HH[i + 2, i], i + 1, i + 2)
        lmul!(G, view(HH, :, i:m))
        HH[i + 2, i] = Htmp
        if i < iend - 2
            Htmp = HH[i + 3, i + 1]
            HH[i + 3, i + 1] = 0
        end
        rmul!(view(HH, 1:min(i + 3, iend), :), G')
        Z === nothing || rmul!(Z, G')
    end
    return HH
end

const _STANDARDIZE_DEFAULT = true

function _gschur!(H::HessenbergFactorization{T}, Z=nothing;
                  tol = eps(real(T)), debug = false, shiftmethod = :Francis,
                  maxiter = 100*size(H, 1), standardize = _STANDARDIZE_DEFAULT,
                  kwargs...) where {T <: Real}
    n = size(H, 1)
    istart = 1
    iend = n
    HH = H.data
    τ = Rotation(Givens{T}[])
    w = Vector{Complex{T}}(undef, n)
    iwcur = n
    function putw!(w,z)
        w[iwcur] = z
        iwcur -= 1
    end

    # iteration count
    i = 0

    @inbounds while true
        i += 1
        if i > maxiter
            throw(ArgumentError("iteration limit $maxiter reached"))
        end

        # Determine if the matrix splits. Find lowest positioned subdiagonal "zero"
        for _istart in iend - 1:-1:1
            if abs(HH[_istart + 1, _istart]) < tol*(abs(HH[_istart, _istart]) + abs(HH[_istart + 1, _istart + 1]))
                    istart = _istart + 1
                if T <: Real
                    debug && @printf("Split! Subdiagonal element is: %10.3e and istart now %6d\n", HH[istart, istart - 1], istart)
                    standardize && (HH[istart, istart-1] = 0) # clean
                else
                    debug && @printf("Split! Subdiagonal element is: %10.3e%+10.3eim and istart now %6d\n", reim(HH[istart, istart - 1])..., istart)
                end
                break
            elseif _istart > 1 && abs(HH[_istart, _istart - 1]) < tol*(abs(HH[_istart - 1, _istart - 1]) + abs(HH[_istart, _istart]))
                if T <: Real
                    debug && @printf("Split! Next subdiagonal element is: %10.3e and istart now %6d\n", HH[_istart, _istart - 1], _istart)
                else
                    debug && @printf("Split! Next subdiagonal element is: %10.3e%+10.3eim and istart now %6d\n", reim(HH[_istart, _istart - 1])..., _istart)
                end
                istart = _istart
                break
            end
            istart = 1
        end

        # if block size is one we deflate
        if istart >= iend
            debug && @printf("Bottom deflation! Block size is one. New iend is %6d\n", iend - 1)
            standardize && putw!(w,HH[iend,iend])
            iend -= 1

        # and the same for a 2x2 block
        elseif istart + 1 == iend
            debug && @printf("Bottom deflation! Block size is two. New iend is %6d\n", iend - 2)

            if standardize
                debug && println("std. iend = $iend")
                H2 = HH[iend-1:iend,iend-1:iend]
                G2,w1,w2 = _gs2x2!(H2,iend)
                putw!(w,w2)
                putw!(w,w1)
                lmul!(G2,view(HH,:,istart:n))
                rmul!(view(HH,1:iend,:),G2')
                HH[iend-1:iend,iend-1:iend] .= H2 # clean
                if iend > 2
                    HH[iend-1,iend-2] = 0
#                    HH[iend-2,iend-1] = 0
                end
                Z === nothing || rmul!(Z,G2')
            end
            iend -= 2

        # run a QR iteration
        # shift method is specified with shiftmethod kw argument
        else
            Hmm = HH[iend, iend]
            Hm1m1 = HH[iend - 1, iend - 1]
            d = Hm1m1*Hmm - HH[iend, iend - 1]*HH[iend - 1, iend]
            t = Hm1m1 + Hmm
            t = iszero(t) ? eps(real(one(t))) : t # introduce a small pertubation for zero shifts
            if T <: Real
                debug && @printf("block start is: %6d, block end is: %6d, d: %10.3e, t: %10.3e\n", istart, iend, d, t)
            else
                debug && @printf("block start is: %6d, block end is: %6d, d: %10.3e%+10.3eim, t: %10.3e%+10.3eim\n", istart, iend, reim(d)..., reim(t)...)
            end

            if shiftmethod == :Francis
                # Run a bulge chase
                if iszero(i % 10)
                    # Vary the shift strategy to avoid dead locks
                    # We use a Wilkinson-like shift as suggested in "Sandia technical report 96-0913J: How the QR algorithm fails to converge and how fix it".

                    if T <: Real
                        debug && @printf("Wilkinson-like shift! Subdiagonal is: %10.3e, last subdiagonal is: %10.3e\n", HH[iend, iend - 1], HH[iend - 1, iend - 2])
                    else
                        debug && @printf("Wilkinson-like shift! Subdiagonal is: %10.3e%+10.3eim, last subdiagonal is: %10.3e%+10.3eim\n", reim(HH[iend, iend - 1])..., reim(HH[iend - 1, iend - 2])...)
                    end
                    _d = t*t - 4d

                    if _d isa Real && _d >= 0
                        # real eigenvalues
                        a = t/2
                        b = sqrt(_d)/2
                        s = a > Hmm ? a - b : a + b
                    else
                        # complex case
                        s = t/2
                    end
                    singleShiftQR!(HH, τ, Z, s, istart, iend)
                else
                    # most of the time use Francis double shifts
                    if T <: Real
                        debug && @printf("Francis double shift! Subdiagonal is: %10.3e, last subdiagonal is: %10.3e\n", HH[iend, iend - 1], HH[iend - 1, iend - 2])
                    else
                        debug && @printf("Francis double shift! Subdiagonal is: %10.3e%+10.3eim, last subdiagonal is: %10.3e%+10.3eim\n", reim(HH[iend, iend - 1])..., reim(HH[iend - 1, iend - 2])...)
                    end
                    doubleShiftQR!(HH, τ, Z, t, d, istart, iend)
                end
            elseif shiftmethod == :Rayleigh
                if T <: Real
                    debug && @printf("Single shift with Rayleigh shift! Subdiagonal is: %10.3e\n", HH[iend, iend - 1])
                else
                    debug && @printf("Single shift with Rayleigh shift! Subdiagonal is: %10.3e%+10.3eim\n", reim(HH[iend, iend - 1])...)
                end

                # Run a bulge chase
                singleShiftQR!(HH, τ, Z, Hmm, istart, iend)
            else
                throw(ArgumentError("only support supported shift methods are :Francis (default) and :Rayleigh. You supplied $shiftmethod"))
            end
        end
        if iend <= 2
            if standardize
                if iend == 1
                    putw!(w,HH[iend,iend])
                elseif iend == 2
                    debug && println("final std. iend = $iend")
                    H2 = HH[1:2,1:2]
                    G2,w1,w2 = _gs2x2!(H2,2)
                    putw!(w,w2)
                    putw!(w,w1)
                    lmul!(G2,HH)
                    rmul!(view(HH,1:2,:),G2')
                    HH[1:2,1:2] .= H2 # clean
                    Z === nothing || rmul!(Z,G2')
                end
            end
            break
        end
    end

    TT = triu(HH,-1)
    if standardize
        v = w
    else
        v = _geigvals!(TT)
    end
    return Schur{T,typeof(TT)}(TT, Z === nothing ? similar(TT,0,0) : Z, v)
end

# compute Schur decomposition of real 2x2 in standard form
# return corresponding Givens and eigenvalues
# Translated from LAPACK::dlanv2
function _gs2x2!(H2::StridedMatrix{T},jj) where {T <: Real}
    a,b,c,d = H2[1,1], H2[1,2], H2[2,1], H2[2,2]
    sgn(x) = (x < 0) ? -one(T) : one(T) # fortran sign differs from Julia
    half = one(T) / 2
    small = 4eps(T) # how big discriminant must be for easy reality check
    if c==0
        cs = one(T)
        sn = zero(T)
    elseif b==0
        # swap rows/cols
        cs = zero(T)
        sn = one(T)
        a,b,c,d = d,-c,zero(T),a
    elseif ((a-d) == 0) && (b*c < 0)
        # nothing to do
        cs = one(T)
        sn = zero(T)
    else
        asubd = a-d
        p = half*asubd
        bcmax = max(abs(b),abs(c))
        bcmis = min(abs(b),abs(c)) * sgn(b) * sgn(c)
        scale = max(abs(p), bcmax)
        z = (p / scale) * p + (bcmax / scale) * bcmis
        # if z is of order machine accuracy: postpone decision
        if z >= small
            # real eigenvalues
            z = p + sqrt(scale) * sqrt(z) * sgn(p)
            a = d + z
            d -= (bcmax / z) * bcmis
            τ = hypot(c,z)
            cs = z / τ
            sn = c / τ
            b -= c
            c = zero(T)
        else
            # complex or almost equal real eigenvalues
            σ = b + c
            τ = hypot(σ, asubd)
            cs = sqrt(half * (one(T) + abs(σ) / τ))
            sn = -(p / (τ * cs)) * sgn(σ)
            # apply rotations
            aa = a*cs + b*sn
            bb = -a*sn + b*cs
            cc = c*cs + d*sn
            dd = -c*sn + d*cs
            a = aa*cs + cc*sn
            b = bb*cs + dd*sn
            c = -aa*sn + cc*cs
            d = -bb*sn + dd*cs
            midad = half * (a+d)
            a = midad
            d = a
            if (c != 0)
                if (b != 0)
                    if b*c >= 0
                        # real eigenvalues
                        sab = sqrt(abs(b))
                        sac = sqrt(abs(c))
                        p = sab*sac*sgn(c)
                        τ = one(T) / sqrt(abs(b+c))
                        a = midad + p
                        d = midad - p
                        b -= c
                        c = 0
                        cs1 = sab*τ
                        sn1 = sac*τ
                        cs, sn = cs*cs1 - sn*sn1, cs*sn1 + sn*cs1
                    end
                else
                    b,c = -c,zero(T)
                    cs,sn = -sn,cs
                end
            end
        end
    end

    if c==0
        w1,w2 = a,d
    else
        rti = sqrt(abs(b))*sqrt(abs(c))
        w1 = a + rti*im
        w2 = d - rti*im
    end
    H2[1,1], H2[1,2], H2[2,1], H2[2,2] = a,b,c,d
    G = Givens(jj-1,jj,cs,sn)
    return G,w1,w2
end

function gschur!(A::StridedMatrix{T}; wantZ::Bool=true, scale::Bool=true,
                 permute::Bool=false, kwargs...) where {T <: Real}
    n = checksquare(A)
    # permute &&
    #    throw(ArgumentError("permute option is not available for this method"))
    if scale
        scaleA, cscale, anrm = _scale!(A)
    else
        scaleA = false
    end
    H = _hessenberg!(A)
    if wantZ
        Z = Matrix{T}(I, n, n)
        for j=n-1:-1:1
            lmul!(H.τ[j], view(Z, j+1:n, j:n))
            Z[1:j-1,j] .= 0
        end
        S = _gschur!(H, Z; kwargs...)
    else
        S = _gschur!(H; kwargs...)
    end
    if scaleA
        safescale!(S.T, cscale, anrm)
        safescale!(S.values, cscale, anrm)
    end
    S
end

function singleShiftQR!(HH::StridedMatrix{T}, τ::Rotation, Z, shift::Number, istart::Integer, iend::Integer) where {T <: Real}
    m = size(HH, 1)
    H11 = HH[istart, istart]
    H21 = HH[istart + 1, istart]
    if m > istart + 1
        Htmp = HH[istart + 2, istart]
        HH[istart + 2, istart] = 0
    end
    G, _ = givens(H11 - shift, H21, istart, istart + 1)
    lmul!(G, view(HH, :, istart:m))
    rmul!(view(HH, 1:min(istart + 2, iend), :), G')
    lmul!(G, τ)
    Z === nothing || rmul!(Z,G')
    for i = istart:iend - 2
        G, _ = givens(HH[i + 1, i], HH[i + 2, i], i + 1, i + 2)
        lmul!(G, view(HH, :, i:m))
        HH[i + 2, i] = Htmp
        if i < iend - 2
            Htmp = HH[i + 3, i + 1]
            HH[i + 3, i + 1] = 0
        end
        rmul!(view(HH, 1:min(i + 3, iend), :), G')
        Z === nothing || rmul!(Z,G')
    end
    return HH
end

function doubleShiftQR!(HH::StridedMatrix{T}, τ::Rotation, Z, shiftTrace::Number, shiftDeterminant::Number, istart::Integer, iend::Integer) where {T <: Real}
    m = size(HH, 1)
    H11 = HH[istart, istart]
    H21 = HH[istart + 1, istart]
    Htmp11 = HH[istart + 2, istart]
    HH[istart + 2, istart] = 0
    if istart + 3 <= m
        Htmp21 = HH[istart + 3, istart]
        HH[istart + 3, istart] = 0
        Htmp22 = HH[istart + 3, istart + 1]
        HH[istart + 3, istart + 1] = 0
    else
        # values doen't matter in this case but variables should be initialized
        Htmp21 = Htmp22 = Htmp11
    end
    G1, r = givens(H11*H11 + HH[istart, istart + 1]*H21 - shiftTrace*H11 + shiftDeterminant, H21*(H11 + HH[istart + 1, istart + 1] - shiftTrace), istart, istart + 1)
    G2, _ = givens(r, H21*HH[istart + 2, istart + 1], istart, istart + 2)
    vHH = view(HH, :, istart:m)
    lmul!(G1, vHH)
    lmul!(G2, vHH)
    vHH = view(HH, 1:min(istart + 3, m), :)
    rmul!(vHH, G1')
    rmul!(vHH, G2')
    lmul!(G1, τ)
    lmul!(G2, τ)
    Z === nothing || rmul!(Z,G1')
    Z === nothing || rmul!(Z,G2')
    for i = istart:iend - 2
        for j = 1:2
            if i + j + 1 > iend break end
            # G, _ = givens(H.H,i+1,i+j+1,i)
            G, _ = givens(HH[i + 1, i], HH[i + j + 1, i], i + 1, i + j + 1)
            lmul!(G, view(HH, :, i:m))
            HH[i + j + 1, i] = Htmp11
            Htmp11 = Htmp21
            # if i + j + 2 <= iend
                # Htmp21 = HH[i + j + 2, i + 1]
                # HH[i + j + 2, i + 1] = 0
            # end
            if i + 4 <= iend
                Htmp22 = HH[i + 4, i + j]
                HH[i + 4, i + j] = 0
            end
            rmul!(view(HH, 1:min(i + j + 2, iend), :), G')
            Z === nothing || rmul!(Z,G')
        end
    end
    return HH
end

# get eigenvalues from a quasitriangular Schur factor
function _geigvals!(HH::StridedMatrix{T}; tol = eps(T)) where {T <: Real}
    # this is used for the non-standard form
    n = size(HH, 1)
    vals = Vector{complex(T)}(undef, n)
    i = 1
    while i < n
        Hii = HH[i, i]
        Hi1i1 = HH[i + 1, i + 1]
        rtest = tol*(abs(Hi1i1) + abs(Hii))
        if abs(HH[i + 1, i]) < rtest
            vals[i] = Hii
            i += 1
        else
            d = Hii*Hi1i1 - HH[i, i + 1]*HH[i + 1, i]
            t = Hii + Hi1i1
            x = 0.5*t
            y = sqrt(complex(x*x - d))
            vals[i] = x + y
            vals[i + 1] = x - y
            i += 2
        end
    end
    if i == n
        vals[i] = HH[n, n]
    end
    return vals
end

include("vectors.jl")
include("triang.jl")

include("norm1est.jl")
include("sylvester.jl")
include("ordschur.jl")

end # module
