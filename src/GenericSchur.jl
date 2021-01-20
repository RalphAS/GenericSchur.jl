module GenericSchur
using LinearAlgebra
using LinearAlgebra: Givens, Rotation
using Printf
import LinearAlgebra: lmul!, mul!, checksquare, ldiv!

# This is the public interface of the package.
# Wrappers like `schur` and `eigvals` should just work.
import LinearAlgebra: schur!, eigvals!, eigvecs, eigen!
export triangularize, eigvalscond, subspacesep, balance!

schur!(A::StridedMatrix{T}; kwargs...) where {T} = gschur!(A; kwargs...)

function eigvals!(A::StridedMatrix{T}; kwargs...) where {T}
    S = gschur!(A; wantZ=false, kwargs...)
    S.values
end

# This is probably the best we can do unless LinearAlgebra coöperates
"""
    eigvecs(S::Schur{<:Complex}; left=false) => Matrix

Compute right or left eigenvectors from a Schur decomposition.
Eigenvectors are returned as columns of a matrix.
The returned eigenvectors have unit Euclidean norm, and the largest
elements are real.
"""
function eigvecs(S::Schur{T}; left::Bool=false) where {T <: Complex}
    if left
        v = _gleigvecs!(S.T,S.Z)
    else
        v = _geigvecs!(S.T,S.Z)
    end
    _enormalize!(v)
    v
end
function _enormalize!(v::AbstractMatrix{T}) where T
    n = size(v,1)
    for j=1:n
        s = one(real(T)) / norm(v[:,j],2)
        t = abs2(v[1,j])
        i0 = 1
        for i=2:n
            u = abs2(v[i,j])
            if  u > t
                i0 = i
                t = u
            end
        end
        t = s * conj(v[i0,j]) / sqrt(t)
        for i=1:n
            v[i,j] *=  t
        end
        v[i0,j] = real(v[i0,j])
    end
end

if VERSION < v"1.2-"
    eigsortby = nothing
else
    using LinearAlgebra: eigsortby, sorteig!
end

function eigen!(A::StridedMatrix{T}; permute=true, scale=true,
                sortby::Union{Function,Nothing}=eigsortby) where {T <: Real}
    if permute || scale
        A, B = balance!(A, scale=scale, permute=permute)
    end
    S = triangularize(schur(A))
    v = _geigvecs!(S.T,S.Z)
    if permute || scale
        lmul!(B, v)
    end
    _enormalize!(v)
    if sortby !== nothing
        return LinearAlgebra.Eigen(sorteig!(S.values, v, sortby)...)
    else
        return LinearAlgebra.Eigen(S.values,v)
    end
end

function eigen!(A::StridedMatrix{T}; permute::Bool=true, scale::Bool=true,
                sortby::Union{Function,Nothing}=eigsortby) where {T <: Complex}
    if permute || scale
        A, B = balance!(A, scale=scale, permute=permute)
    end
    S = schur(A)
    v = _geigvecs!(S.T,S.Z)
    if permute || scale
        lmul!(B, v)
    end
    _enormalize!(v)
    if sortby !== nothing
        return LinearAlgebra.Eigen(sorteig!(S.values, v, sortby)...)
    else
        return LinearAlgebra.Eigen(S.values,v)
    end
end

############################################################################
# Internal implementation follows

macro mydebug(expr); nothing; end

function _fmt_nr(z::Complex)
    s = @sprintf("%10.3e%+10.3eim", reim(z)...)
    return s
end
function _fmt_nr(x::Real)
    s = @sprintf("%10.3e", x)
    return s
end

include("util.jl")
include("lapack_extras.jl")
include("hessenberg.jl")

if VERSION < v"1.3"
    _getdata(H::HessenbergFactorization) = H.data
    const HessenbergArg = HessenbergFactorization
else
    _getdata(H::Hessenberg) = H.H.data
    const HessenbergArg = Hessenberg
end

include("householder.jl")
include("balance.jl")

#
# portions translated from LAPACK::zlahqr
# LAPACK Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
function _gschur!(H::HessenbergArg{T}, Z=nothing;
                 maxiter = 100*size(H, 1), maxinner = 30*size(H, 1), kwargs...
                 ) where {T <: Complex}
    n = size(H, 1)
    istart = 1
    iend = n
    w = Vector{T}(undef, n)
    HH = _getdata(H)

    RT = real(T)
    ulp = eps(RT)
    smallnum = safemin(RT) * (n / ulp)
    rzero = zero(RT)
    half = 1 / RT(2)
    threeq = 3 / RT(4)

    for j=1:n-1
        HH[j+2:n,j] .= zero(T)
    end

    # iteration count
    it = 0

    @inbounds while iend >= 1
        # key for comparison to zlahqr:
        # I => iend
        # L => istart
        # K => _istart + 1
        istart = 1
        for its=0:maxinner
            it += 1
            if it > maxiter
                throw(ArgumentError("iteration limit $maxiter reached"))
            end

            # Determine if the matrix splits.
            # Find lowest positioned subdiagonal "zero" if any; reset istart if found.
            for _istart in iend-1:-1:istart
                if abs1(HH[_istart+1, _istart]) <= smallnum
                    istart = _istart + 1
                    @mydebug println("Split1! at istart = $istart, subdiag is ",
                                     _fmt_nr(HH[istart, istart-1]))
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
                        @mydebug println("Split2! at istart = $istart, subdiag is ",
                                         _fmt_nr(HH[istart, istart-1]))
                        break
                    end
                end
            end # check for split

            if istart > 1
                # clean up
                HH[istart, istart-1] = zero(T)
            end

            # if block size is one we deflate
            if istart >= iend
                iend -= 1
                @mydebug println("Bottom deflation! Block size is one. New iend is $iend")
                break # from "inner" loop
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
                # Wilkinson's shift
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
            @mydebug println("block range $(istart):$(iend) shift ", _fmt_nr(t))

            # following zlahqr, only use single-shift
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
                 kwargs...) where T <: Complex
    n = checksquare(A)
    if scale
        scaleA, cscale, anrm = _scale!(A)
    else
        scaleA = false
    end
    H = _hessenberg!(A)
    if wantZ
        Z = _materializeQ(H)
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

function singleShiftQR!(HH::StridedMatrix{T}, Z, shift::Number, istart::Integer, iend::Integer) where {T <: Complex}
    n = size(HH, 1)
    ulp = eps(real(eltype(HH)))

    @mydebug Hsave = Z*HH*Z'
    function dcheck(str)
        print(str," decomp err ", norm(Hsave - Z*HH*Z'))
    end

    # key:
    # istart => L
    # iend => I
    # istart1 => M

    # look for two consecutive small subdiagonals
    istart1 = -1
    h11s = zero(eltype(HH))
    h21 = zero(real(eltype(HH)))
    flag = false
    v = zeros(T,2)
    @inbounds for mm = iend-1:-1:istart+1
        # determine the effect of starting the single-shift Francis
        # iteration at row mm: see if this would make HH[mm,mm-1] tiny.
        h11 = HH[mm,mm]
        h22 = HH[mm+1,mm+1]
        h11s = h11 - shift
        h21 = real(HH[mm+1,mm]) # for reflector
        s = abs1(h11s) + abs(h21)
        h11s /= s
        h21 /= s
        v .= (h11s, T(h21))
        h10 = real(HH[mm,mm-1])
        if abs(h10) * abs(h21) <= ulp * (abs1(h11s) * (abs1(h11) + abs1(h22)))
            istart1 = mm
            flag = true
            break
        end
    end
    @inbounds begin
      if !flag
        istart1 = istart
        h11 = HH[istart,istart]
        h22 = HH[istart+1,istart+1]
        h11s = h11 - shift
        h21 = real(HH[istart+1,istart])
        s = abs1(h11s) + abs(h21)
        h11s /= s
        h21 /= s
        v .= (h11s,T(h21))
      end
    end

    @inbounds for k = istart1:iend-1
        if k > istart1
            # prepare to chase bulge
            v .= (HH[k,k-1], HH[k+1,k-1])
        # else
        #    use v from above to create a bulge
        end
        τ1 = LinearAlgebra.reflector!(v)
        if k > istart1
            HH[k,k-1] = v[1]
            HH[k+1,k-1] = zero(T)
        end
        v2 = v[2]
        τ2 = real(τ1*v2)
        # lmul!(R,view(HH,:,k:n))
        for j = k:n
            ss = τ1' * HH[k,j] + τ2 * HH[k+1,j]
            HH[k,j] -= ss
            HH[k+1,j] -= ss * v2
        end
        # rmul!(view(HH,1:min(iend,k+2),:),R)
        for j = 1:min(k+2,iend)
            ss = τ1 * HH[j,k] + τ2 * HH[j,k+1]
            HH[j,k] -= ss
            HH[j,k+1] -= ss * v2'
        end
        if !(Z === nothing)
            # rmul!(Z,R)
            for j = 1:n
                ss = τ1 * Z[j,k] + τ2 * Z[j,k+1]
                Z[j,k] -= ss
                Z[j,k+1] -= ss * v2'
            end
        end
        if (k == istart1) && (istart1 > istart)
            # If the QR step started at a row below istart because two consecutive
            # small subdiagonals were found, extra scaling must be performed
            # to ensure reality of HH[istart1,istart1-1]
            t = 1 - τ1
            t /= abs(t)
            HH[istart1+1,istart1] *= t'
            if istart1 + 2 <= iend
                HH[istart1+2, istart1+1] *= t
            end
            for j = istart1:iend
                if j != istart1 + 1
                    if n > j
                        HH[j,j+1:n] *= t
                    end
                    HH[1:j-1,j] *= t'
                    if !(Z === nothing)
                        Z[1:n, j] *= t'
                    end
                end
            end
        end
        @mydebug dcheck(" QR k=$k")
    end
    # ensure reality of tail
    @inbounds begin
      t = HH[iend,iend-1]
      if imag(t) != 0
        rt = abs(t)
        HH[iend, iend-1] = rt
        t /= rt
        if n > iend
            HH[iend, iend+1:n] *= t'
        end
        HH[1:iend-1, iend] *= t
        if !(Z === nothing)
            Z[1:n, iend] *= t
        end
      end
    end
    @mydebug dcheck(" QR post")
    @mydebug println()
    return HH
end

const _STANDARDIZE_DEFAULT = true

# Mostly copied from GenericLinearAlgebra
function _gschur!(H::HessenbergArg{T}, Z=nothing;
                  tol = eps(real(T)), shiftmethod = :Francis,
                  maxiter = 100*size(H, 1), standardize = _STANDARDIZE_DEFAULT,
                  kwargs...) where {T <: Real}
    n = size(H, 1)
    istart = 1
    iend = n
    HH = _getdata(H)
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
                @mydebug println("Split1! at $istart, subdiagonal element is ",
                                 _fmt_nr(HH[istart, istart-1]))
                standardize && (HH[istart, istart-1] = 0) # clean
                break
            elseif _istart > 1 && abs(HH[_istart, _istart - 1]) < tol*(abs(HH[_istart - 1, _istart - 1]) + abs(HH[_istart, _istart]))
                istart = _istart
                @mydebug println("Split2! at $istart, subdiagonal element is ",
                                 _fmt_nr(HH[_istart, _istart-1]))
                break
            end
            istart = 1
        end

        # if block size is one we deflate
        if istart >= iend

            standardize && putw!(w,HH[iend,iend])
            iend -= 1
            @mydebug println("Bottom deflation! Block size is one. New iend is $iend")
        # and the same for a 2x2 block
        elseif istart + 1 == iend
            @mydebug println("Bottom deflation! Block size is two. New iend is $(iend-2)")

            if standardize
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
            if iszero(i % 10)
                # Use Eispack exceptional shift
                β = abs(HH[iend, iend - 1]) + abs(HH[iend - 1, iend - 2])
                d = (Hmm + β)^2 - Hmm*β/2
                t = 2*Hmm + 3*β/2
            else
                d = Hm1m1*Hmm - HH[iend, iend - 1]*HH[iend - 1, iend]
                t = Hm1m1 + Hmm
            end
            # t = iszero(t) ? eps(real(one(t))) : t # introduce a small pertubation for zero shifts
            @mydebug println("block range: $(istart):$(iend) d: ", _fmt_nr(d),
                             " t: ", _fmt_nr(t))

            if shiftmethod == :Francis
                # Run a bulge chase
                if iszero(i % 10)
                    # Vary the shift strategy to avoid dead locks
                    # We use a Wilkinson-like shift as suggested in "Sandia technical report 96-0913J: How the QR algorithm fails to converge and how fix it".

                    @mydebug println("Wilkinson-like shift! Subdiagonal is: ",
                                     _fmt_nr(HH[iend, iend - 1]),
                                     " last subdiagonal is: ",
                                     _fmt_nr(HH[iend-1, iend-2]))
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
                    @mydebug println("Francis double shift! Subdiagonal is: ",
                                     _fmt_nr(HH[iend, iend - 1]),
                                     " last subdiagonal is: ",
                                     _fmt_nr(HH[iend - 1, iend - 2]))
                    doubleShiftQR!(HH, τ, Z, t, d, istart, iend)
                end
            elseif shiftmethod == :Rayleigh
                @mydebug println("Single shift with Rayleigh shift! Subdiagonal is: ",
                                 _fmt_nr(HH[iend, iend - 1]))

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
                    @mydebug println("final std. iend = $iend")
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
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
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
                 kwargs...) where {T <: Real}
    n = checksquare(A)
    if scale
        scaleA, cscale, anrm = _scale!(A)
    else
        scaleA = false
    end
    H = _hessenberg!(A)
    if wantZ
        Z = _materializeQ(H)
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

include("generalized.jl")
include("gsylvester.jl")

include("ordschur.jl")
include("gcondition.jl")

end # module
