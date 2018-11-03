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

# completely unsafe upper triangular solve
# NB: works with leading n×n block of A and leading n terms of b, x
function _usolve0!(A::StridedMatrix{T}, n, x, cnorm) where {T}
    @inbounds for j in n:-1:1
        xj = x[j] = A[j,j] \ x[j]
        for i in j-1:-1:1
            x[i] -= A[i,j] * xj
        end
    end
    # for future compatibility: overflow-safe version would return scale
    one(real(T))
end

# index-unsafe upper triangular solve
# with scaling to prevent overflow
# Actually solves A[1:n,1:n] * x = s * b and returns s, a scaling factor
# such that components of x are manageable.
# NB: works with leading n×n block of A and leading n terms of b, x
# Hence all the loops must be explicit.
# This is a translation of (part of) LAPACK::zlatrs, q.v. for details
function _usolve!(A::StridedMatrix{T}, n, x, cnorm) where {T}

    cabs2(z) = abs(real(z)/2) + abs(imag(z)/2)

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
        tscale = one(RT) / (smallnum * tmax)
        cnorm .*= tscale
    end
    # bound on solution vector:
    xmax = cabs2(x[1])
    @inbounds for j=2:n
        xmax = max(xmax,cabs2(x[j]))
    end
    xbound = xmax
    if tscale != one(RT)
        grow = zero(RT)
    else
        # compute grow
        grow = half / max(xbound, smallnum)
        xbound = grow
        for j=n:-1:1
            # give up if too small
            (grow <= smallnum) && break
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
        grow = xbound
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
            xscale = bignum * half / xmax
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
                    rec = tjj * bignum / xj
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
    end
    if tscale != one(RT)
        @inbounds for i=1:n; cnorm[i] *= one(RT)/tscale; end
    end
    xscale
end
