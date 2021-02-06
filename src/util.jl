using LinearAlgebra: BlasComplex, BlasFloat, BlasReal
"""
a "safe" version of `floatmin`, such that `1/sfmin` does not overflow.
"""
function safemin(T)
    sfmin = floatmin(T)
    small = one(T) / floatmax(T)
    if small >= sfmin
        sfmin = small * (one(T) + eps(T))
    end
    sfmin
end

function _scale!(A::AbstractArray{T}) where {T}
    smlnum = sqrt(safemin(real(T))) / eps(real(T))
    bignum = 1 / smlnum
    anrm = norm(A,Inf)
    scaleA = false
    cscale = one(real(T))
    if anrm > 0 && anrm < smlnum
        scaleA = true
        cscale = smlnum
    elseif anrm > bignum
        scaleA = true
        cscale = bignum
    end
    scaleA && safescale!(A, anrm, cscale)
    scaleA, cscale, anrm
end

abs1(z) = abs(z)
abs1(z::T) where {T <: Complex} = abs(real(z))+abs(imag(z))

# translated from xLASCL (LAPACK)
"""
multiply an array `A` by a real number `cto/cfrom` expressed as a ratio,
computing without over/underflow where possible.

`cfrom` must not be zero.
"""
function safescale!(A::AbstractArray{T}, cfrom::Real, cto::Real) where T
    # CHECKME: what are correct type constraints on cfrom,cto?
    isnan(cto) && throw(ArgumentError("cto must be a valid number"))
    (isnan(cfrom) || (cfrom == 0)) && throw(ArgumentError("cfrom must be a valid nonzero number"))
    smlnum = safemin(real(T))
    bignum = 1/smlnum
    cfromc = cfrom
    ctoc = cto
    done = false
    while !done
        cfrom1 = cfromc * smlnum
        if cfrom1 == cfromc
            # cfromc is ±∞; multiplier is signed 0 or NaN
            mul = ctoc / cfromc
            done = true
            cto1 = ctoc
        else
            cto1 = ctoc / bignum
            if cto1 == ctoc
                # ctoc is 0 or ±∞, so is correct multiplier
                mul = cto
                done = true
                cfromc = one(real(T))
            elseif abs(cfrom1) > abs(ctoc) && (ctoc != 0)
                mul = smlnum
                done = false
                cfromc = one(real(T))
            elseif abs(cto1) > abs(cfromc)
                mul = bignum
                done = false
                cfromc = cfrom1
            else
                mul = ctoc / cfromc
                done = true
            end
        end
        A .= A .* mul
    end
end

# Rank one update

## General
### BLAS
rankUpdate!(α::T, x::StridedVector{T}, y::StridedVector{T}, A::StridedMatrix{T}) where {T<:BlasReal} = BLAS.ger!(α, x, y, A)

### Generic
function rankUpdate!(α::Number, x::StridedVector, y::StridedVector, A::StridedMatrix)
    m, n = size(A, 1), size(A, 2)
    m == length(x) || throw(DimensionMismatch("x vector has wrong length"))
    n == length(y) || throw(DimensionMismatch("y vector has wrong length"))
    for j = 1:n
        yjc = y[j]'
        for i = 1:m
            A[i,j] += x[i]*α*yjc
        end
    end
end

# index-unsafe upper triangular solve
# with scaling to prevent overflow
# Actually solves A[1:n,1:n] * x = s * b and returns s, a scaling factor
# such that components of x are manageable.
# NB: works with leading n×n block of A and leading n terms of b, x
# Hence all the loops must be explicit.
# This is a translation of (part of) LAPACK::zlatrs, q.v. for details.
# Provide `b` in argument `x`; note that `cnorm` may be updated.
function _usolve!(A::StridedMatrix{T}, n, x, cnorm) where {T}

    cabs1half(z) = abs(real(z)/2) + abs(imag(z)/2)

    RT = real(T)
    half = one(RT) / 2
    smallnum = safemin(RT) / eps(RT)
    bignum = one(RT)/ smallnum
    xscale = one(RT)

    tmax = abs(cnorm[1])
    @inbounds for j=2:n
        tmax = max(tmax, abs(cnorm[j]))
    end

    if tmax <= bignum * half
        tscale = one(RT)
    else
        tscale = half / (smallnum * tmax)
        cnorm[1:n] .*= tscale
    end
    # bound on solution vector:
    xmax = cabs1half(x[1])
    @inbounds for j=2:n
        xmax = max(xmax,cabs1half(x[j]))
    end
    xbound = xmax
    if tscale != one(RT)
        grow = zero(RT)
    else
        # compute grow
        grow = half / max(xbound, smallnum)
        xbound = grow
        toosmall = false
        for j=n:-1:1
            # give up if too small
            toosmall = (grow <= smallnum)
            toosmall && break
            tjjs = A[j,j]
            tjj = abs1(tjjs)
            if tjj >= smallnum
                xbound = min(xbound, min(one(RT), tjj)*grow)
            else
                # M[j] could overflow
                xbound = zero(RT)
            end
            if tjj + cnorm[j] >= smallnum
                grow *= (tjj / (tjj + cnorm[j]))
            else
                # grow could overflow, clamp
                grow = zero(RT)
            end
        end
        if !toosmall
            grow = xbound
        end
    end

    if grow * tscale > smallnum
        # bound is ok; use standard arithmetic
        @inbounds for j in n:-1:1
            xj = x[j] = A[j,j] \ x[j]
            for i in j-1:-1:1
                x[i] -= A[i,j] * xj
            end
        end
    else
        if xmax > bignum * half
            # scale so all(abs.(x) .<= bignum)
            xscale = (bignum * half) / xmax
            @inbounds for i=1:n; x[i] *= xscale; end
            xmax = bignum
        else
            xmax *= 2
        end
        for j in n:-1:1
            # compute x[j] = b[j] / A[j,j], scaling if necessary
            xj = abs1(x[j])
            tjjs = A[j,j] * tscale
            tjj = abs1(tjjs)
            if tjj > smallnum
                # abs(A[j,j] > smallnum):
                if tjj < one(RT)
                    if xj > tjj*bignum
                        # scale x by 1/b[j]
                        rec = one(RT) / xj
                        @inbounds for i=1:n; x[i] *= rec; end
                        xscale *= rec
                        xmax *= rec
                    end
                end
                x[j] = x[j] / tjjs
                xj = abs1(x[j])
            elseif tjj > 0
                # 0 < abs(A[j,j]) <= smallnum
                if xj > tjj*bignum
                    # scale by (1/abs(x[j]))*abs(A[j,j]))*bignum
                    # to avoid overflow when dividing by A[j,j]
                    rec = (tjj * bignum) / xj
                    if cnorm[j] > one(RT)
                        # scale by 1/cnorm[j] to avoid overflow
                        # when multiplying x[j] by column j
                        rec /= cnorm[j]
                    end
                    @inbounds for i=1:n; x[i] *= rec; end
                    xscale *= rec
                    xmax *= rec
                end
                x[j] = x[j] / tjjs # FIXME: use zladiv scheme
                xj = abs1(x[j])
            else
                # A[j,j] = 0; set x to eⱼ, xscale to 0
                # and compute a null vector.
                @inbounds for i=1:n; x[i] = zero(T); end
                x[j] = one(T)
                xscale = zero(RT)
                xmax = zero(RT)
            end
            # scale x if necc. to avoid overflow when adding multiple of A[:,j]
            if xj > one(RT)
                rec = one(RT) / xj
                if cnorm[j] > (bignum - xmax) * rec
                    # scale by 1/(2abs(x[j]))
                    rec *= half
                    @inbounds for i=1:n; x[i] *= rec; end
                    xscale *= rec
                end
            elseif (xj * cnorm[j] > bignum - xmax)
                # scale by 1/2
                @inbounds for i=1:n; x[i] *= half; end
                xscale *= half
            end
            if j > 1
                # compute update
                xjt = x[j] * tscale
                @inbounds for i=1:j-1
                    x[i] -= xjt * A[i,j]
                end
                xmax = abs1(x[1])
                @inbounds for i=2:j-1
                    xmax = max(xmax, abs1(x[i]))
                end
            end
        end
        xscale /= tscale
    end
    if tscale != one(RT)
        @inbounds for i=1:n; cnorm[i] *= one(RT)/tscale; end
    end
    return xscale
end

# conjugate transpose version
function _cusolve!(A::StridedMatrix{T}, n, x, cnorm) where {T}

    cabs1half(z) = abs(real(z)/2) + abs(imag(z)/2)

    RT = real(T)
    half = one(RT) / 2
#    smallnum = safemin(RT) / (2*eps(RT)) # WARNING: assumes base 2
    smallnum = safemin(RT) / eps(RT)
    bignum = one(RT)/ smallnum
    xscale = one(RT)

    tmax = abs(cnorm[1])
    @inbounds for j=2:n
        tmax = max(tmax, abs(cnorm[j]))
    end

    if tmax <= bignum * half
        tscale = one(RT)
    else
        tscale = half / (smallnum * tmax)
        cnorm[1:n] .*= tscale
    end
    # bound on solution vector:
    xmax = cabs1half(x[1])
    @inbounds for j=2:n
        xmax = max(xmax,cabs1half(x[j]))
    end
    xbound = xmax
    if tscale != one(RT)
        grow = zero(RT)
    else
        # compute grow
        grow = half / max(xbound, smallnum)
        xbound = grow
        toosmall = false
        for j=1:n
            # give up if too small
            toosmall = (grow <= smallnum)
            toosmall && break
            # G[j] = max(G[j-1], M[j-1]*(1+cnorm[j]))
            xj = one(RT) + cnorm[j]
            grow = min(grow, xbound / xj)
            tjjs = A[j,j]
            tjj = abs1(tjjs)
            if tjj >= smallnum
                if xj > tjj
                    xbound *= (tjj / xj)
                end
            else
                # M[j] could overflow
                xbound = zero(RT)
            end
        end
        if !toosmall
            grow = min(grow, xbound)
        end
    end

    if grow * tscale > smallnum
        # bound is ok; use standard arithmetic
        @inbounds for j in 1:n
            z = x[j]
            for i in 1:j-1
                z -= conj(A[i,j]) * x[i]
            end
            x[j] = conj(A[j,j]) \ z
        end
    else
        if xmax > bignum * half
            # scale so all(abs.(x) .<= bignum)
            xscale = (bignum * half) / xmax
            @inbounds for i=1:n; x[i] *= xscale; end
            xmax = bignum
        else
            xmax *= 2
        end
        for j=1:n
            # compute x[j] = b[j] - Σ_(k≠j) A[k,j] x[k]
            xj = abs1(x[j])
            uscale = complex(tscale)
            rec = one(RT) / max(xmax, one(RT))
            if cnorm[j] > (bignum - xj) * rec
                # if x[j] could overflow scale x by 1/(2 xmax)
                rec *= half
                tjjs = conj(A[j,j]) * tscale
                tjj = abs1(tjjs)
                if tjj > one(RT)
                    # divide by A[j,j] when scaling if A[j,j] > 1
                    rec = min(one(RT), rec*tjj)
                    uscale /= tjjs
                end
                if rec < one(RT)
                    @inbounds for i=1:n; x[i] *= rec; end
                    xscale *= rec
                    xmax *= rec
                end
            end
            csumj = zero(T)
            @inbounds for i=1:j-1
                csumj += (conj(A[i,j]) * uscale) * x[i]
            end
            if uscale == complex(tscale)
                # if diagonal wasn't used to scale
                # compute x[j] = (x[j] - csumj) / A[j,j]
                x[j] -= csumj
                xj = abs1(x[j])
                tjjs = conj(A[j,j]) * tscale

                # compute x[j] /= A[j,j], scaling if necessary
                tjj = abs1(tjjs)
                if tjj > smallnum
                    if tjj < one(RT)
                        if xj > tjj * bignum
                            rec = one(RT) / xj
                            @inbounds for i=1:n; x[i] *= rec; end
                            xscale *= rec
                            xmax *= rec
                        end
                    end
                    x[j] = x[j] / tjjs
                elseif tjj > zero(RT)
                    # tiny diag
                    if xj > tjj * bignum
                        rec = (tjj * bignum) / xj
                        @inbounds for i=1:n; x[i] *= rec; end
                        xscale *= rec
                        xmax *= rec
                    end
                    x[j] = x[j] / tjjs
                else
                    # zero diag: compute a null vector of Aᴴ
                    x[1:n] .= zero(T)
                    x[j] = one(T)
                    xscale = zero(RT)
                    xmax = zero(RT)
                end
            else
                # if dot product was already divided by A[j,j]
                # compute x[j] = x[j] / A[j,j] - csumj
                x[j] = x[j] / tjjs - csumj
            end
            xmax = max(xmax, abs1(x[j]))
        end
        xscale /= tscale
    end
    if tscale != one(RT)
        @inbounds for i=1:n; cnorm[i] *= one(RT)/tscale; end
    end
    xscale
end

# update scale and sumsq s.t. (scale^2 * sumsq) is increased by ∑ |x[j]|^2
# scale is a running quasi-inf-norm.
# based on LAPACK::zlassq
function _ssq(x::AbstractVector{T}, scale, sumsq) where {T}
    n = length(x)
    rone = one(real(T))
    for ix=1:n
        t1 = abs(real(x[ix]))
        if t1 > 0 || isnan(t1)
            if scale < t1
                sumsq = rone + sumsq * (scale / t1)^2
                scale = t1
            else
                sumsq += (t1 / scale)^2
            end
        end
        if T <: Complex
            t1 = abs(imag(x[ix]))
            if t1 > 0 || isnan(t1)
                if scale < t1
                    sumsq = rone + sumsq * (scale / t1)^2
                    scale = t1
                else
                    sumsq += (t1 / scale)^2
                end
            end
        end
    end
    return scale, sumsq
end

# stdlib norm2 uses a poorly implemented generic scheme for short vectors.
function _norm2(x::AbstractVector{T}) where {T<:Real}
    require_one_based_indexing(x)
    n = length(x)
    n < 1 && return zero(T)
    n == 1 && return abs(x[1])
    scale = zero(T)
    ssq = zero(T)
    for xi in x
        if !iszero(xi)
            a = abs(xi)
            if scale < a
                ssq = one(T) + ssq * (scale / a)^2
                scale = a
            else
                ssq += (a / scale)^2
            end
        end
    end
    return scale * sqrt(ssq)
end

function _norm2(x::AbstractVector{T}) where {T<:Complex}
    require_one_based_indexing(x)
    n = length(x)
    RT = real(T)
    n < 1 && return zero(RT)
    n == 1 && return abs(x[1])
    scale = zero(RT)
    ssq = zero(RT)
    for xx in x
        xr,xi = reim(xx)
        if !iszero(xr)
            a = abs(xr)
            if scale < a
                ssq = one(RT) + ssq * (scale / a)^2
                scale = a
            else
                ssq += (a / scale)^2
            end
        end
        if !iszero(xi)
            a = abs(xi)
            if scale < a
                ssq = one(RT) + ssq * (scale / a)^2
                scale = a
            else
                ssq += (a / scale)^2
            end
        end
    end
    return scale * sqrt(ssq)
end

# As of v1.5, Julia hypot() w/ >2 args is unprotected (the documentation lies),
# so we need this.
# translation of dlapy3, assuming NaN propagation
function _hypot3(x::T, y::T, z::T) where {T}
    xa = abs(x)
    ya = abs(y)
    za = abs(z)
    w = max(xa, ya, za)
    rw = one(real(T)) / w
    r::real(T) = w * sqrt((rw * xa)^2 + (rw * ya)^2 + (rw * za)^2)
    return r
end

if VERSION < v"1.2"
    function require_one_based_indexing(A::AbstractArray)
        if Base.has_offset_axes(A)
            throw(ArgumentError("offset axes are not supported"))
        end
    end
end
