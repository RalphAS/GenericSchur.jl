# This file is part of GenericSchur.jl, released under the MIT "Expat" license
# Portions derived from LAPACK, see below.

# Eigenvectors

"""
`_geigvecs!(T[,Z])`

Compute right eigenvectors of a complex upper triangular matrix `T`.
If another matrix `Z` is provided, multiply by it to get eigenvectors of `Zᵀ T Z`.
Typically `T` and `Z` are components of a Schur decomposition.
Temporarily mutates `T`.
"""
function _geigvecs!(TT::StridedMatrix{T},
                    Z::StridedMatrix{T}=Matrix{T}(undef,0,0)
                    ) where {T <: Complex}
    # based on LAPACK::ztrevc
    # Copyright:
    # Univ. of Tennessee
    # Univ. of California Berkeley
    # Univ. of Colorado Denver
    # NAG Ltd.
    n = size(TT,1)
    RT = real(T)
    ulp = eps(RT)
    # Note: LAPACK has
    #   smallnum = safemin(RT) * (n / ulp)
    # but that makes no sense to me and breaks some tests
    smallnum = safemin(RT) * n
    vectors = Matrix{T}(undef,n,n)
    v = zeros(T,n)

    # save diagonal since we modify it to avoid copies
    tdiag = diag(TT)

    # We use the 1-norms of the strictly upper part of TT columns
    # to avoid overflow
    tnorms = zeros(RT,n)
    @inbounds for j=2:n
        for i=1:j-1
            tnorms[j] += abs(TT[i,j])
        end
    end

    for ki=n:-1:1
        smin = max(ulp * abs1(TT[ki,ki]), smallnum)
        #
        # (T[1:k,1:k]-λI) x = b
        # where k=kᵢ-1

        v[1] = one(T) # for ki=1
        @inbounds for k=1:ki-1
            v[k] = -TT[k,ki]
        end
        @inbounds for k=1:ki-1
            TT[k,k] -= TT[ki,ki]
            (abs1(TT[k,k]) < smin) && (TT[k,k] = smin)
        end
        if ki > 1
            vscale = _usolve!(TT,ki-1,v,view(tnorms, 1:ki-1))
            v[ki] = vscale
        else
            vscale = one(RT)
        end
        if size(Z,1) > 0
            # This is done here to avoid allocating a work matrix
            # and to exploit the subspace property to reduce work.
            # Using a work matrix would allow for level-3 ops (cf. ztrevc3).
            @inbounds for j=1:n
                vectors[j,ki] = vscale * Z[j,ki]
                for i=1:ki-1
                    vectors[j,ki] += Z[j,i] * v[i]
                end
            end
        else
            @inbounds for j=1:ki
                vectors[j,ki] = v[j]
            end
            vectors[ki+1:n,ki] .= zero(T)
        end

        # normalize
        t0 = abs1(vectors[1,ki])
        @inbounds for i=2:n; t0 = max(t0, abs1(vectors[i,ki])); end
        remax = one(RT) / t0
        @inbounds for i=1:n; vectors[i,ki] *= remax; end

        # restore diagonal
        @inbounds for k=1:ki-1
            TT[k,k] = tdiag[k]
        end
    end

    vectors
end

"""
`_gleigvecs!(T[,Z])`

Compute left eigenvectors of a complex upper triangular matrix `T`.
If another matrix `Z` is provided, multiply by it to get eigenvectors of `Zᵀ T Z`.
Typically `T` and `Z` are components of a Schur decomposition.
Temporarily mutates `T`.
"""
function _gleigvecs!(TT::StridedMatrix{T},
                    Z::StridedMatrix{T}=Matrix{T}(undef,0,0)
                    ) where {T <: Complex}
    # based on LAPACK::ztrevc
    # Copyright:
    # Univ. of Tennessee
    # Univ. of California Berkeley
    # Univ. of Colorado Denver
    # NAG Ltd.
    n = size(TT,1)
    RT = real(T)
    ulp = eps(RT)
    # replaces LAPACK's smallnum = safemin(RT) * (n / ulp)
    smallnum = safemin(RT) * n
    vectors = Matrix{T}(undef,n,n)
    v = zeros(T,n)

    # save diagonal since we modify it to avoid copies
    tdiag = diag(TT)

    # we use the 1-norm of strictly upper part of TT
    # cols to control overflow
    tnorms = zeros(RT,n)
    @inbounds for j=2:n
        for i=1:j-1
            tnorms[j] += abs1(TT[i,j])
        end
    end

    # logic from ztrevc
    for ki=1:n
        smin = max(ulp * abs1(TT[ki,ki]), smallnum)

        # form RHS
        @inbounds for k=ki+1:n
            v[k] = -conj(TT[ki,k])
        end
        v[1:ki] .= zero(T)
        # solve (T[ki+1:n,ki+1:n] - T[ki,ki])ᴴ x = σ v
        @inbounds for k=ki+1:n
            TT[k,k] -= TT[ki,ki]
            (abs1(TT[k,k]) < smin) && (TT[k,k] = smin)
        end
        if ki < n
            vscale = _cusolve!(view(TT,ki+1:n,ki+1:n),n-ki,view(v,ki+1:n),tnorms)
            v[ki] = vscale
        else
            v[n] = one(T)
            vscale = one(RT)
        end
        if size(Z,1) > 0
            # This is done here to avoid allocating a work matrix
            # and to exploit the subspace property to reduce work.
            # Using a work matrix would allow for level-3 ops (cf. ztrevc3).
            @inbounds for j=1:n
                vectors[j,ki] = vscale * Z[j,ki]
                for i=ki+1:n
                    vectors[j,ki] += Z[j,i] * v[i]
                end
            end
        else
            @inbounds for j=ki:n
                vectors[j,ki] = v[j]
            end
            vectors[1:ki-1,ki] .= zero(T)
        end

        # normalize
        t0 = abs1(vectors[1,ki])
        @inbounds for i=2:n; t0 = max(t0, abs1(vectors[i,ki])); end
        remax = one(RT) / t0
        @inbounds for i=1:n; vectors[i,ki] *= remax; end

        # restore diagonal
        @inbounds for k=ki+1:n
            TT[k,k] = tdiag[k]
        end
    end

    vectors
end

"""
`_geigvecs(S, T [,Z])`

Compute right eigenvectors of a matrix pair `S, T` where `S` is [quasi-]upper triangular
and `T` is upper triangular.
If another matrix `Z` is provided, multiply by it to get eigenvectors of
`Qᵀ S Z` and `Qᵀ T Z`.
Typically `S`, `T` and `Z` are components of a generalized Schur decomposition.
"""
function _geigvecs end

# right eigenvectors, complex generalized case
function _geigvecs(S::StridedMatrix{T}, P::StridedMatrix{T},
                    Z::StridedMatrix{T}=Matrix{T}(undef,0,0)
                    ) where {T <: Complex}
    # translated from LAPACK::ztgevc
    # Copyright:
    # Univ. of Tennessee
    # Univ. of California Berkeley
    # Univ. of Colorado Denver
    # NAG Ltd.
    n = size(S,1)
    RT = real(T)
    ulp = eps(RT)
    safmin = safemin(RT)
    smallnum = safmin * (n / ulp)
    bigx = one(RT) / smallnum
    bignum = one(RT) / (n * safmin)

    vectors = Matrix{T}(undef,n,n)
    v = zeros(T,n)
    if size(Z,1) > 0
        v2 = zeros(T,n)
    end

    # We use the 1-norms of the strictly upper part of S, P columns
    # to avoid overflow
    anorm = abs1(S[1,1])
    bnorm = abs1(P[1,1])
    snorms = zeros(RT,n)
    pnorms = zeros(RT,n)
    @inbounds for j=2:n
        for i=1:j-1
            snorms[j] += abs1(S[i,j])
            pnorms[j] += abs1(P[i,j])
        end
        anorm = max(anorm,snorms[j] + abs1(S[j,j]))
        bnorm = max(bnorm,pnorms[j] + abs1(P[j,j]))
    end
    ascale = one(RT) / max(anorm, safmin)
    bscale = one(RT) / max(bnorm, safmin)

    idx = n+1
    for ki=n:-1:1
        idx -= 1
        if abs1(S[ki,ki]) <= safmin && abs(real(P[ki,ki])) <= safmin
            # singular pencil; return unit eigenvector
            vectors[:,idx] .= zero(T)
            vectors[idx,idx] = one(T)
        else
            # compute coeffs a,b in (a A - b B) x = 0
            t = 1 / max(abs1(S[ki,ki]) * ascale,
                        abs(real(P[ki,ki])) * bscale, safmin)
            sα = (t * S[ki,ki]) * ascale
            sβ = (t * real(P[ki,ki])) * bscale
            acoeff = sβ * ascale
            bcoeff = sα * bscale
            # scale to avoid underflow

            lsa = abs(sβ) >= safmin && abs(acoeff) < smallnum
            lsb = abs1(sα) >= safmin && abs1(bcoeff) < smallnum
            s = one(RT)
            if lsa
                s = (smallnum / abs(sβ)) * min(anorm, bigx)
            end
            if lsb
                s = max(s, (smallnum / abs1(sα)) * min(bnorm, bigx))
            end
            if lsa || lsb
                s = min(s, 1 / (safmin * max( one(RT), abs(acoeff),
                                              abs1(bcoeff))))
                if lsa
                    acoeff = ascale * (s * sβ)
                else
                    acoeff *= s
                end
                if lsb
                    bcoeff = bscale * (s * sα)
                else
                    bcoeff *= s
                end
            end
            aca = abs(acoeff)
            acb = abs1(bcoeff)
            xmax = one(RT)
            v .= zero(T)
            dmin = max(ulp * aca * anorm, ulp * acb * bnorm, safmin)

            # triangular solve of (a A - b B) x = 0, columnwise
            # v[1:j-1] contains sums w
            # v[j+1:ki] contains x

            v[1:ki-1] .= acoeff * S[1:ki-1,ki] - bcoeff * P[1:ki-1,ki]
            v[ki] = one(T)

            for j=ki-1:-1:1
                # form x[j] = -v[j] / d
                # with scaling and perturbation
                d = acoeff * S[j,j] - bcoeff * P[j,j]
                if abs1(d) <= dmin
                    d = complex(dmin)
                end
                if abs1(d) < one(RT)
                    if abs1(v[j]) >= bignum * abs1(d)
                        t = 1 / abs1(v[j])
                        v[1:ki] *= t
                    end
                end
                v[j] = -v[j] / d
                if j > 1
                    # w = w + x[j] * (a S[:,j] - b P[:,j]) with scaling

                    if abs1(v[j]) > one(RT)
                        t = 1 / abs1(v[j])
                        if aca * snorms[j] + acb * pnorms[j] >= bignum * t
                            v[1:ki] *= t
                        end
                    end
                    ca = acoeff * v[j]
                    cb = bcoeff * v[j]
                    v[1:j-1] += ca * S[1:j-1,j] - cb * P[1:j-1,j]
                end
            end # for j (solve loop)
            if size(Z,1) > 0
                mul!(v2, Z, v)
                v, v2 = v2, v
                iend = n
            else
                iend = ki
            end

            xmax = zero(RT)
            for jr=1:iend
                xmax = max(xmax, abs1(v[jr]))
            end
            if xmax > safmin
                t = 1 / xmax
                vectors[1:iend,idx] .= t * v[1:iend]
            else
                iend = 0
            end
            vectors[iend+1:n,idx] .= zero(T)
        end # nonsingular branch
    end # index loop

    return vectors
end # function

