module GenericSchur

# These are introduced here.
export triangularize, eigvalscond, subspacesep, balance!

VERSION >= v"1.11.0-DEV.469" && eval(
    Meta.parse(
        """
        public
        gschur!,
        ggschur!,
        geigen!,
        geigvals!,
        gordschur!,
        geigvecs,
        UnconvergedException,
        IllConditionException,
        Balancer
        """
    )
)

const STypes = Union{AbstractFloat, Complex{T} where {T <: AbstractFloat}}

using LinearAlgebra
using LinearAlgebra: eigsortby, sorteig!, checksquare, givensAlgorithm, _ordschur!
using LinearAlgebra: RealHermSymComplexHerm, Algorithm, QRIteration, Givens
using Printf

using Base: require_one_based_indexing

"""
    UnconvergedException

Exception thrown when an iterative algorithm does not converge within the allowed
number of steps.
"""
struct UnconvergedException <: Exception
    msg::String
end
function Base.showerror(io::IO, e::UnconvergedException)
    return print(io, "Convergence failure; $(e.msg)")
end

"""
    IllConditionException

Exception thrown when argument matrix or matrices are too ill-conditioned
for the requested operation. The `index` field may indicate the block
where near-singularity was detected.
"""
struct IllConditionException <: Exception
    index::Int
end

# placeholder for future options
const piracy = true

include("pirates.jl")

############################################################################
# Internal implementations follow

"""
geigen!(A, alg=QRIteration(); sortby=eigsortby) -> E::Eigen

Computes the eigen-decomposition of a Hermitian (or real symmetric) matrix `A`
using a generic implementation.
Currently `alg` may be `QRIteration()` or `DivideAndConquer()`.
Otherwise, similar to `LinearAlgebra.eigen!`.
"""
function geigen!(
        A::RealHermSymComplexHerm{<:STypes, <:StridedMatrix},
        alg::Algorithm = QRIteration();
        sortby::Union{Function, Nothing} = eigsortby
    )
    H = hessenberg!(A)
    V = _materializeQ(H)
    S = gschur!(H, V, alg)
    λ = S.values
    return LinearAlgebra.Eigen(sorteig!(λ, V, sortby)...)
end

function geigvals!(
        A::RealHermSymComplexHerm{<:STypes, <:StridedMatrix},
        alg::Algorithm = QRIteration();
        sortby::Union{Function, Nothing} = eigsortby
    )
    H = hessenberg!(A)
    S = gschur!(H, nothing, alg)
    λ = S.values
    return sorteig!(λ, sortby)
end

# fallback methods; only some algorithms have selection capability

function geigen!(
        A::RealHermSymComplexHerm{<:STypes, <:StridedMatrix},
        irange::UnitRange,
        alg::Algorithm = QRIteration(); kwargs...
    )
    throw(ArgumentError("eigenvalue selection is not implemented for $alg"))
end
function geigen!(
        A::RealHermSymComplexHerm{<:STypes, <:StridedMatrix},
        vl::Real, vu::Real,
        alg::Algorithm = QRIteration(); kwargs...
    )
    throw(ArgumentError("eigenvalue selection is not implemented for $alg"))
end

# debugging stuff
macro mydebug(expr)
    return nothing
end

function _fmt_nr(z::Complex)
    s = @sprintf("%10.3e%+10.3eim", reim(z)...)
    return s
end
function _fmt_nr(x::Real)
    s = @sprintf("%10.3e", x)
    return s
end

include("util.jl")
include("hessenberg.jl")

_getdata(H::Hessenberg) = H.H.data

include("householder.jl")
include("balance.jl")

#
# portions translated from LAPACK::zlahqr
# LAPACK Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
function gschur!(
        H::Hessenberg{Complex{RT}}, Z = nothing;
        maxiter = 100 * size(H, 1), maxinner = 30 * size(H, 1),
        checksd = true, kwargs...
    ) where {RT <: AbstractFloat}
    T = Complex{RT}
    n = size(H, 1)
    istart = 1
    iend = n
    w = Vector{T}(undef, n)
    HH = _getdata(H)

    if checksd
        for j in 1:(n - 1)
            isreal(HH[j + 1, j]) || throw(ArgumentError("algorithm assumes real subdiagonal"))
        end
    end

    ulp = eps(RT)
    smallnum = safemin(RT) * (n / ulp)
    rzero = zero(RT)
    half = 1 / RT(2)
    threeq = 3 / RT(4)
    v2 = zeros(T, 2)

    for j in 1:(n - 1)
        HH[(j + 2):n, j] .= zero(T)
    end

    # iteration count
    it = 0

    @inbounds while iend >= 1
        # key for comparison to zlahqr:
        # I => iend
        # L => istart
        # K => _istart + 1
        istart = 1
        for its in 0:maxinner
            it += 1
            if it > maxiter
                throw(UnconvergedException("iteration limit $maxiter reached"))
            end

            # Determine if the matrix splits.
            # Find lowest positioned subdiagonal "zero" if any; reset istart if found.
            for _istart in (iend - 1):-1:istart
                if abs1(HH[_istart + 1, _istart]) <= smallnum
                    istart = _istart + 1
                    @mydebug println(
                        "Split1! at istart = $istart, subdiag is ",
                        _fmt_nr(HH[istart, istart - 1])
                    )
                    break
                end
                # deflation criterion from Ahues & Tisseur (LAWN 122, 1997)
                tst = abs1(HH[_istart, _istart]) + abs1(HH[_istart + 1, _istart + 1])
                if tst == 0
                    if (_istart - 1 >= 1)
                        tst += abs(real(HH[_istart, _istart - 1]))
                    end
                    if (_istart + 2 <= n)
                        tst += abs(real(HH[_istart + 2, _istart + 1]))
                    end
                end
                if abs(real(HH[_istart + 1, _istart])) <= ulp * tst
                    ab = max(
                        abs1(HH[_istart + 1, _istart]),
                        abs1(HH[_istart, _istart + 1])
                    )
                    ba = min(
                        abs1(HH[_istart + 1, _istart]),
                        abs1(HH[_istart, _istart + 1])
                    )
                    aa = max(
                        abs1(HH[_istart + 1, _istart + 1]),
                        abs1(HH[_istart, _istart] - HH[_istart + 1, _istart + 1])
                    )
                    bb = min(
                        abs1(HH[_istart + 1, _istart + 1]),
                        abs1(HH[_istart, _istart] - HH[_istart + 1, _istart + 1])
                    )
                    s = aa + ab
                    if ba * (ab / s) <= max(smallnum, ulp * (bb * (aa / s)))
                        istart = _istart + 1
                        @mydebug println(
                            "Split2! at istart = $istart, subdiag is ",
                            _fmt_nr(HH[istart, istart - 1])
                        )
                        break
                    end
                end
            end # check for split

            if istart > 1
                # clean up
                HH[istart, istart - 1] = zero(T)
            end

            # if block size is one we deflate
            if istart >= iend
                iend -= 1
                @mydebug println("Bottom deflation! Block size is one. New iend is $iend")
                break # from "inner" loop
            end

            # select shift
            # exceptional shift logic adapted from LAPACK zlahqr
            if its % 30 == 10
                s = threeq * abs(real(HH[istart + 1, istart]))
                t = s + HH[istart, istart]
            elseif its % 30 == 20
                s = threeq * abs(real(HH[iend, iend - 1]))
                t = s + HH[iend, iend]
            else
                # Wilkinson's shift
                t = HH[iend, iend]
                u = sqrt(HH[iend - 1, iend]) * sqrt(HH[iend, iend - 1])
                s = abs1(u)
                if s ≠ rzero
                    x = half * (HH[iend - 1, iend - 1] - t)
                    sx = abs1(x)
                    s = max(s, abs1(x))
                    y = s * sqrt((x / s)^2 + (u / s)^2)
                    if sx > rzero
                        if real(x / sx) * real(y) + imag(x / sx) * imag(y) < rzero
                            y = -y
                        end
                    end
                    t -= u * (u / (x + y))
                end
            end # shift selection

            # run a QR iteration
            # following zlahqr, only use single-shift
            singleShiftQR!(HH, Z, t, istart, iend, v2)

        end # inner loop
    end # outer loop
    w = diag(HH)
    return Schur(triu(HH), Z === nothing ? Matrix{T}(undef, 0, 0) : Z, w)
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
function gschur!(
        A::StridedMatrix{Complex{T}}; wantZ::Bool = true, scale::Bool = true,
        kwargs...
    ) where {T <: AbstractFloat}
    n = checksquare(A)
    if scale
        scaleA, cscale, anrm = _scale!(A)
    else
        scaleA = false
    end
    H = _hessenberg!(A)
    if wantZ
        Z = _materializeQ(H)
        S = gschur!(H, Z; checksd = false, kwargs...)
    else
        S = gschur!(H; checksd = false, kwargs...)
    end
    if scaleA
        safescale!(S.T, cscale, anrm)
        S.values .= diag(S.T, 0)
    end
    return S
end

function singleShiftQR!(HH::StridedMatrix{T}, Z, shift::Number, istart::Integer, iend::Integer, v) where {T <: Complex}
    n = size(HH, 1)
    ulp = eps(real(eltype(HH)))

    @mydebug dcheck = DebugCompare(Z * HH * Z', "decomp error")

    # key:
    # istart => L
    # iend => I
    # istart1 => M

    # look for two consecutive small subdiagonals
    istart1 = -1
    h11s = zero(eltype(HH))
    h21 = zero(real(eltype(HH)))
    flag = false
    @inbounds for mm in (iend - 1):-1:(istart + 1)
        # determine the effect of starting the single-shift Francis
        # iteration at row mm: see if this would make HH[mm,mm-1] tiny.
        h11 = HH[mm, mm]
        h22 = HH[mm + 1, mm + 1]
        h11s = h11 - shift
        h21 = real(HH[mm + 1, mm]) # for reflector
        s = abs1(h11s) + abs(h21)
        h11s /= s
        h21 /= s
        v .= (h11s, T(h21))
        h10 = real(HH[mm, mm - 1])
        if abs(h10) * abs(h21) <= ulp * (abs1(h11s) * (abs1(h11) + abs1(h22)))
            istart1 = mm
            flag = true
            break
        end
    end
    @inbounds begin
        if !flag
            istart1 = istart
            h11 = HH[istart, istart]
            h22 = HH[istart + 1, istart + 1]
            h11s = h11 - shift
            h21 = real(HH[istart + 1, istart])
            s = abs1(h11s) + abs(h21)
            h11s /= s
            h21 /= s
            v .= (h11s, T(h21))
        end
    end
    @mydebug println(
        "QR sweep $(istart1):$(iend) shift ", _fmt_nr(shift),
        " subdiag ", _fmt_nr(HH[iend, iend - 1])
    )

    @inbounds for k in istart1:(iend - 1)
        if k > istart1
            # prepare to chase bulge
            v .= (HH[k, k - 1], HH[k + 1, k - 1])
            # else
            #    use v from above to create a bulge
        end
        τ1 = _reflector!(v)
        # τ1 = LinearAlgebra.reflector!(v)
        if k > istart1
            HH[k, k - 1] = v[1]
            HH[k + 1, k - 1] = zero(T)
        end
        v2 = v[2]
        τ2 = real(τ1 * v2)
        # lmul!(R,view(HH,:,k:n))
        for j in k:n
            ss = τ1' * HH[k, j] + τ2 * HH[k + 1, j]
            HH[k, j] -= ss
            HH[k + 1, j] -= ss * v2
        end
        # rmul!(view(HH,1:min(iend,k+2),:),R)
        for j in 1:min(k + 2, iend)
            ss = τ1 * HH[j, k] + τ2 * HH[j, k + 1]
            HH[j, k] -= ss
            HH[j, k + 1] -= ss * v2'
        end
        if !(Z === nothing)
            # rmul!(Z,R)
            for j in 1:n
                ss = τ1 * Z[j, k] + τ2 * Z[j, k + 1]
                Z[j, k] -= ss
                Z[j, k + 1] -= ss * v2'
            end
        end
        if (k == istart1) && (istart1 > istart)
            # If the QR step started at a row below istart because two consecutive
            # small subdiagonals were found, extra scaling must be performed
            # to ensure reality of HH[istart1,istart1-1]
            t = one(T) - τ1
            t /= abs(t)
            HH[istart1 + 1, istart1] *= t'
            if istart1 + 2 <= iend
                HH[istart1 + 2, istart1 + 1] *= t
            end
            for j in istart1:iend
                if j != istart1 + 1
                    if n > j
                        HH[j, (j + 1):n] *= t
                    end
                    HH[1:(j - 1), j] *= t'
                    if !(Z === nothing)
                        Z[1:n, j] *= t'
                    end
                end
            end
        end
        @mydebug dcheck(Z * HH * Z', " QR k=$k")
    end
    # ensure reality of tail
    @inbounds begin
        t = HH[iend, iend - 1]
        if imag(t) != 0
            rt = abs(t)
            HH[iend, iend - 1] = rt
            t /= rt
            if n > iend
                HH[iend, (iend + 1):n] *= t'
            end
            HH[1:(iend - 1), iend] *= t
            if !(Z === nothing)
                Z[1:n, iend] *= t
            end
        end
    end
    #    @mydebug dcheck(Z*HH*Z', " QR post")
    @mydebug println()
    return HH
end

# based on LAPACK dlahqr
"""
gschur!(H::Hessenberg, Z) -> F::Schur

Compute the Schur decomposition of a Hessenberg matrix.  Subdiagonals of `H` must be real.
If `Z` is provided, it is updated with the unitary transformations of the decomposition.
"""
function gschur!(
        H::Hessenberg{T}, Z::Union{Nothing, AbstractMatrix} = nothing;
        tol = eps(real(T)),
        maxiter = 100 * size(H, 1), standardize::Union{Nothing, Bool} = nothing,
        kwargs...
    ) where {T <: AbstractFloat}
    if !(standardize === nothing)
        @warn "obsolete keyword `standardize` in gschur!" maxlog = 1
    end
    n = size(H, 1)
    if !(Z === nothing)
        size(Z, 2) == n || throw(DimensionMismatch("second dimension of Z must match H"))
    end
    istart = 1
    iend = n
    HH = _getdata(H)
    triu!(HH, -1)
    w = Vector{Complex{T}}(undef, n)
    iwcur = n
    function putw!(w, z, iw)
        w[iw] = z
        return iw - 1
    end

    smallnum = floatmin(T) * (n / eps(T))
    threeq = 3 / T(4)
    m7_16 = -7 / T(16)
    v3 = zeros(T, 3)
    H2 = zeros(T, 2, 2)

    # iteration count
    iter = 0

    # Main loop.
    @inbounds while iend >= 1
        # eigenvalues in diagonal entries iend+1:n have converged.
        istart = 1
        iterqr = 0
        deflate = false
        while true
            iter += 1
            if iter > maxiter
                throw(UnconvergedException("iteration limit $maxiter reached"))
            end

            # istart => L
            # iend => I

            # Determine if the matrix splits.
            split = false
            for k in iend:-1:(istart + 1)
                if abs(HH[k, k - 1]) < smallnum
                    split = true
                else
                    Hkk = HH[k, k]
                    Hkm1km1 = HH[k - 1, k - 1]
                    t = abs(Hkm1km1) + abs(Hkk)
                    if t == 0
                        if k > 2
                            t += abs(HH[k - 1, k - 2])
                        end
                        if k + 1 <= n
                            t += abs(HH[k + 1, k])
                        end
                    end
                    # conservative deflation criterion from Ahues & Tisseur
                    aHkkm1 = abs(HH[k, k - 1])
                    if aHkkm1 <= t * eps(T)
                        aHkm1k = abs(HH[k - 1, k])
                        ab = max(aHkkm1, aHkm1k)
                        ba = min(aHkkm1, aHkm1k)
                        aa = max(abs(Hkk), abs(Hkm1km1 - Hkk))
                        bb = min(abs(Hkk), abs(Hkm1km1 - Hkk))
                        s = aa + bb
                        if ba * (ab / s) <= max(smallnum, eps(T) * (bb * (aa / s)))
                            split = true
                        end
                    end
                end
                if split
                    istart = k
                    @mydebug println(
                        "Split at $istart, subdiagonal element is ",
                        _fmt_nr(HH[istart, istart - 1])
                    )
                    break
                end
            end
            if !split
                istart = 1
            end
            if istart > 1
                HH[istart, istart - 1] = zero(T)
            end
            if istart >= iend - 1
                deflate = true
                break
            end

            # run a QR iteration
            iterqr += 1
            if iterqr == 10
                # exceptional shift from top
                s = abs(HH[istart + 1, istart]) + abs(HH[istart + 2, istart + 1])
                H11 = threeq * s + HH[istart, istart]
                H12 = m7_16 * s
                H21 = s
                H22 = H11
            elseif iterqr == 20
                # exceptional shift from bottom
                s = abs(HH[iend, iend - 1]) + abs(HH[iend - 1, iend - 2])
                H11 = threeq * s + HH[iend, iend]
                H12 = m7_16 * s
                H21 = s
                H22 = H11
            else
                # Francis' double shift (i.e. 2nd degree Rayleigh quotient)
                H11 = HH[iend - 1, iend - 1]
                H21 = HH[iend, iend - 1]
                H12 = HH[iend - 1, iend]
                H22 = HH[iend, iend]
            end
            s = abs(H11) + abs(H12) + abs(H21) + abs(H22)
            if s == 0
                r1r, r2r, r1i, r2i = zero(T), zero(T), zero(T), zero(T)
            else
                H11 /= s
                H12 /= s
                H21 /= s
                H22 /= s
                tr = (H11 + H22) / T(2)
                d = (H11 - tr) * (H22 - tr) - H12 * H21
                rtd = sqrt(abs(d))
                if d >= 0
                    # conjugate pair
                    r1r = tr * s
                    r2r = r1r
                    r1i = rtd * s
                    r2i = -r1i
                else
                    # duplicate real shifts
                    r1r = tr + rtd
                    r2r = tr - rtd
                    if abs(r1r - H22) <= abs(r2r - H22)
                        r1r *= s
                        r2r = r1r
                    else
                        r2r *= s
                        r1r = r2r
                    end
                    r1i, r2i = zero(T), zero(T)
                end
            end
            doubleShiftQR!(HH, Z, complex(r1r, r1i), complex(r2r, r2i), istart, iend, v3)
        end
        if deflate && (istart >= iend)
            # if block size is one we deflate
            thisw = HH[iend, iend]
            iwcur = putw!(w, thisw, iwcur)
            iend = istart - 1
            @mydebug println("Deflate one. New iend is $iend. w=", _fmt_nr(thisw))
        elseif deflate && (istart + 1 == iend)
            # and the same for a 2x2 block
            copyto!(H2, view(HH, (iend - 1):iend, (iend - 1):iend))
            G2, w1, w2 = _gs2x2!(H2, iend)
            iwcur = putw!(w, w2, iwcur)
            iwcur = putw!(w, w1, iwcur)
            lmul!(G2, view(HH, :, istart:n))
            rmul!(view(HH, 1:iend, :), G2')
            copyto!(view(HH, (iend - 1):iend, (iend - 1):iend), H2) # clean
            if iend > 2
                HH[iend - 1, iend - 2] = 0
            end
            Z === nothing || rmul!(Z, G2')
        end
        iend = istart - 1
        @mydebug begin
            println(
                "Deflate two. New iend is $iend w1=", _fmt_nr(w1),
                " w2=", _fmt_nr(w2)
            )
        end
    end

    TT = triu(HH, -1)
    return _schurtyped(TT, Z === nothing ? similar(TT, 0, 0) : Z, w)
end

# this should not be needed, but at some point it helped for type stability
if VERSION < v"1.8"
    @inline _schurtyped(T::TM, Z::TM, w::Tw) where {TM <: AbstractMatrix{Te}, Tw} where {Te} = Schur{Te, TM}(T, Z, w)
else
    @inline _schurtyped(T::TM, Z::TM, w::Tw) where {TM <: AbstractMatrix{Te}, Tw} where {Te} = Schur{Te, TM, Tw}(T, Z, w)
end

# compute Schur decomposition of real 2x2 in standard form
# return corresponding Givens and eigenvalues
# Translated from LAPACK::dlanv2
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
function _gs2x2!(H2::StridedMatrix{T}, jj) where {T <: Real}
    a, b, c, d = H2[1, 1], H2[1, 2], H2[2, 1], H2[2, 2]
    sgn(x) = (x < 0) ? -one(T) : one(T) # fortran sign differs from Julia
    half = one(T) / 2
    small = 4eps(T) # how big discriminant must be for easy reality check
    if c == 0
        cs = one(T)
        sn = zero(T)
    elseif b == 0
        # swap rows/cols
        cs = zero(T)
        sn = one(T)
        a, b, c, d = d, -c, zero(T), a
    elseif ((a - d) == 0) && (b * c < 0)
        # nothing to do
        cs = one(T)
        sn = zero(T)
    else
        asubd = a - d
        p = half * asubd
        bcmax = max(abs(b), abs(c))
        bcmis = min(abs(b), abs(c)) * sgn(b) * sgn(c)
        scale = max(abs(p), bcmax)
        z = (p / scale) * p + (bcmax / scale) * bcmis
        # if z is of order machine accuracy: postpone decision
        if z >= small
            # real eigenvalues
            z = p + sqrt(scale) * sqrt(z) * sgn(p)
            a = d + z
            d -= (bcmax / z) * bcmis
            τ = hypot(c, z)
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
            aa = a * cs + b * sn
            bb = -a * sn + b * cs
            cc = c * cs + d * sn
            dd = -c * sn + d * cs
            a = aa * cs + cc * sn
            b = bb * cs + dd * sn
            c = -aa * sn + cc * cs
            d = -bb * sn + dd * cs
            midad = half * (a + d)
            a = midad
            d = a
            if (c != 0)
                if (b != 0)
                    if b * c >= 0
                        # real eigenvalues
                        sab = sqrt(abs(b))
                        sac = sqrt(abs(c))
                        p = sab * sac * sgn(c)
                        τ = one(T) / sqrt(abs(b + c))
                        a = midad + p
                        d = midad - p
                        b -= c
                        c = 0
                        cs1 = sab * τ
                        sn1 = sac * τ
                        cs, sn = cs * cs1 - sn * sn1, cs * sn1 + sn * cs1
                    end
                else
                    b, c = -c, zero(T)
                    cs, sn = -sn, cs
                end
            end
        end
    end

    if c == 0
        w1, w2 = a, d
    else
        rti = sqrt(abs(b)) * sqrt(abs(c))
        w1 = a + rti * im
        w2 = d - rti * im
    end
    H2[1, 1], H2[1, 2], H2[2, 1], H2[2, 2] = a, b, c, d
    G = Givens(jj - 1, jj, cs, sn)
    return G, w1, w2
end

function gschur!(
        A::StridedMatrix{T}; wantZ::Bool = true, scale::Bool = true,
        Zarg::Union{Nothing, Matrix{T}} = nothing,
        Zwrk::Union{Nothing, Matrix{T}} = nothing,
        kwargs...
    ) where {T <: AbstractFloat}
    n = checksquare(A)
    if scale
        scaleA, cscale, anrm = _scale!(A)
    else
        scaleA = false
    end
    H = _hessenberg!(A)
    if wantZ
        if Zarg !== nothing
            nz = checksquare(Zarg)
            Z = _materializeQ!(Zarg, H, Zwrk)
        else
            Z = _materializeQ(H)
        end
        S = gschur!(H, Z; kwargs...)
    else
        S = gschur!(H; kwargs...)
    end
    if scaleA
        safescale!(S.T, cscale, anrm)
        safescale!(S.values, cscale, anrm)
    end
    return S
end

function doubleShiftQR!(
        H::StridedMatrix{T}, Z, shift1, shift2,
        istart::Integer, iend::Integer, v
    ) where {T <: Real}
    n = size(H, 1)
    @mydebug dcheck = DebugCompare(Z * H * Z', "decomp error")

    i1 = 1
    i2 = n
    r1r, r1i = reim(shift1)
    r2r, r2i = reim(shift2)
    v .= zero(T)
    # look for two consecutive small subdiagonals
    mx = istart
    @inbounds for m in (iend - 2):-1:istart
        # Determine whether starting double-shift QR iteration
        # at row m would make H[m,m-1] negligible.
        # Use scaling to avoid over/underflow.
        H21s = H[m + 1, m]
        s = abs(H[m, m] - r2r) + abs(r2i) + abs(H21s)
        H21s /= s
        v[1] = H21s * H[m, m + 1] +
            (H[m, m] - r1r) * ((H[m, m] - r2r) / s) - r1i * (r2i / s)
        v[2] = H21s * (H[m, m] + H[m + 1, m + 1] - r1r - r2r)
        v[3] = H21s * H[m + 2, m + 1]
        s = abs(v[1]) + abs(v[2]) + abs(v[3]) # sum(abs.(v))
        v ./= s
        if (m > istart) && (
                abs(H[m, m - 1]) * (abs(v[2]) + abs(v[3])) <=
                    eps(T) * abs(v[1]) *
                    (abs(H[m - 1, m - 1]) + abs(H[m, m]) + abs(H[m + 1, m + 1]))
            )
            mx = m
            break
        end
    end
    @mydebug println(
        "QR sweep $mx:$iend, shifts ", _fmt_nr(shift1), " ", _fmt_nr(shift2),
        " subdiags ", _fmt_nr(H[iend - 1, iend - 2]), " ", _fmt_nr(H[iend, iend - 1])
    )
    @inbounds for k in mx:(iend - 1)
        # first iteration creates a bulge using reflection based on v
        # subsequent iterations use reflections to restore Hessenberg form
        # in column k-1
        nr = min(3, iend - k + 1) # order of G
        if k > mx
            # v[1:nr] .= H[k:k+nr-1,k-1] # allocates
            for ii in 0:(nr - 1)
                v[1 + ii] = H[k + ii, k - 1]
            end
        end
        @mydebug println(" nr=$nr v=$v ")
        τ1 = _reflector!(view(v, 1:nr))
        if k > mx
            H[k, k - 1] = v[1]
            H[k + 1, k - 1] = zero(T)
            if k < iend - 1
                H[k + 2, k - 1] = zero(T)
            end
        elseif mx > istart
            # avoid problems when v[2] and v[3] underflow
            H[k, k - 1] *= (one(T) - τ1)
        end
        v2 = v[2]
        τ2 = τ1 * v2
        if nr == 3
            v3 = v[3]
            τ3 = τ1 * v3
            # apply G from left; transform H[:,k:i2]
            for j in k:i2
                ss = H[k, j] + v2 * H[k + 1, j] + v3 * H[k + 2, j]
                H[k, j] -= ss * τ1
                H[k + 1, j] -= ss * τ2
                H[k + 2, j] -= ss * τ3
            end
            # apply G from right; transform H[i1:min(k+3,iend),:]
            for j in i1:min(k + 3, iend)
                ss = H[j, k] + v2 * H[j, k + 1] + v3 * H[j, k + 2]
                H[j, k] -= ss * τ1
                H[j, k + 1] -= ss * τ2
                H[j, k + 2] -= ss * τ3
            end
            if !(Z === nothing)
                for j in 1:size(Z, 1)
                    ss = Z[j, k] + v2 * Z[j, k + 1] + v3 * Z[j, k + 2]
                    Z[j, k] -= ss * τ1
                    Z[j, k + 1] -= ss * τ2
                    Z[j, k + 2] -= ss * τ3
                end
            end
        elseif nr == 2
            # apply G from left; transform H[:,k:i2]
            for j in k:i2
                ss = H[k, j] + v2 * H[k + 1, j]
                H[k, j] -= ss * τ1
                H[k + 1, j] -= ss * τ2
            end
            # apply G from right; transform H[i1:iend,:]
            for j in i1:iend
                ss = H[j, k] + v2 * H[j, k + 1]
                H[j, k] -= ss * τ1
                H[j, k + 1] -= ss * τ2
            end
            if !(Z === nothing)
                for j in 1:size(Z, 1)
                    ss = Z[j, k] + v2 * Z[j, k + 1]
                    Z[j, k] -= ss * τ1
                    Z[j, k + 1] -= ss * τ2
                end
            end
        end
        @mydebug dcheck(Z * H * Z', " QR k=$k ")
    end # k loop
    return @mydebug println()
end

include("vectors.jl")
include("triang.jl")

include("symtridiag.jl")
include("symtri_dc.jl")

include("norm1est.jl")
include("sylvester.jl")

include("generalized.jl")
include("rgeneralized.jl")
include("gsylvester.jl")

include("rordschur.jl")
include("ordschur.jl")
include("gcondition.jl")

end # module
