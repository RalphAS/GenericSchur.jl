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

"""
`_gleigvecs(S, T [,Z])`

Compute left eigenvectors of a matrix pair `S, T` where `S` is [quasi-]upper triangular
and `T` is upper triangular.
If another matrix `Z` is provided, multiply by it to get eigenvectors of
`Qᵀ S Z` and `Qᵀ T Z`.
Typically `S`, `T` and `Z` are components of a generalized Schur decomposition.
"""
function _gleigvecs end

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

# right eigenvectors of real generalized Schur
function _geigvecs(S::StridedMatrix{Ty}, P::StridedMatrix{Ty},
                    Z::StridedMatrix{Ty}=Matrix{Ty}(undef,0,0)
) where {Ty <: Real}
    # translated from LAPACK::dtgevc
    # Copyright:
    # Univ. of Tennessee
    # Univ. of California Berkeley
    # Univ. of Colorado Denver
    # NAG Ltd.
    n = size(S,1)
    ulp = eps(Ty)
    safmin = safemin(Ty)
    smallnum = safmin * (n / ulp)
    bigx = one(Ty) / smallnum
    bignum = one(Ty) / (n * safmin)
    safety = Ty(100)

    vectors = Matrix{Complex{Ty}}(undef,n,n)
    v = zeros(Complex{Ty},n)
    # v is WORK(2n+1:4n)

    # We use the 1-norms of the strictly upper part of S, P columns
    # to avoid overflow
    Anorm = abs(S[1,1])
    Bnorm = abs(P[1,1])
    Snorms = zeros(Ty,n)
    Pnorms = zeros(Ty,n)
    # Snorms and Pnorms are WORK[1:n] and WORK[n+1:2n]
    @inbounds for j=2:n
        for i=1:j-1
            Snorms[j] += abs(S[i,j])
            Pnorms[j] += abs(P[i,j])
        end
        Anorm = max(Anorm,Snorms[j] + abs(S[j,j]))
        Bnorm = max(Bnorm,Pnorms[j] + abs(P[j,j]))
    end
    ascale = one(Ty) / max(Anorm, safmin)
    bscale = one(Ty) / max(Bnorm, safmin)
    idx = n+1
    inpair = false
    for je=n:-1:1
        if inpair
            inpair = false
            continue
        end
        nw = 1
        if je > 1
            if S[je,je-1] != 0
                inpair = true
                nw = 2
            end
        end
        if !inpair
            if abs(S[je,je]) <= safmin && abs(P[je,je]) <= safmin
                # singular pencil; return unit eigenvector
                idx -= 1
                vectors[:,idx] .= zero(T)
                vectors[idx,idx] = one(T)
                continue
            end

            # real eigenvalue
            # compute coeffs a,b in (a A - b B) x = 0
            t = 1 / max(abs(S[je,je]) * ascale,
                       abs(P[je,je]) * bscale, safmin)
            sαr = (t * S[je,je]) * ascale
            sβ = (t * P[je,je]) * bscale
            a = sβ * ascale
            br = sαr * bscale
            bi = zero(Ty)

            # scale to avoid underflow

            lsa = abs(sβ) >= safmin && abs(a) < smallnum
            lsb = abs(sαr) >= safmin && abs(br) < smallnum
            s = one(Ty)
            if lsa
                s = (smallnum / abs(sβ)) * min(Anorm, bigx)
            end
            if lsb
                s = max(s, (smallnum / abs(sαr)) * min(Bnorm, bigx))
            end
            if lsa || lsb
                s = min(s, 1 / (safmin * max( one(Ty), abs(a),
                                            abs(br))))
                if lsa
                    a = ascale * (s * sβ)
                else
                    a *= s
                end
                if lsb
                    br = bscale * (s * sαr)
                else
                    br *= s
                end
            end
            aa = abs(a)
            ba = abs(br)
            v .= zero(Complex{Ty})
            v[je] = one(Ty)
            xmax = one(Ty)
            for jr in 1:je-1
                v[jr] = br * P[jr,je] - a * S[jr,je]
            end
        else
            # complex eigenvalue
            scale1, scale2, wr1, wr2, wi = _ggs_2x2(view(S,je-1:je,je-1:je),
                                                   view(P,je-1:je,je-1:je),safmin*safety)
            a = scale1
            br = wr1
            bi = wi
            if bi == 0
                throw(ErrorException("algorithm error: real eigval encountered "
                                     * lazy"when complex expected at $je"))
            end
            # scale to avoid over/underflow
            aa = abs(a)
            ba = abs(br) + abs(bi)
            scale = one(Ty)
            if aa * ulp < safmin && aa >= safmin
                scale = (safmin / ulp) / aa
            end
            if ba * ulp < safmin && ba >= safmin
                scale = max(scale, (safmin / ulp) / ba)
            end
            if safmin * aa > ascale
                scale = ascale / (safmin * aa)
            end
            if safmin * ba > bscale
                scale = min(scale, bscale / (safmin * ba))
            end
            if scale != one(Ty)
                a *= scale
                aa = abs(a)
                br *= scale
                bi *= scale
                ba = abs(br) + abs(bi)
            end
            v .= zero(Ty)
            t1 = a * S[je,je-1]
            t2r = a * S[je,je] - br * P[je,je]
            t2i = -bi * P[je,je]
            if abs(t1) > abs(t2r) + abs(t2i)
                v[je] = one(Ty)
                v[je-1] = - (t2r + im * t2i)/ t1
            else
                v[je-1] = one(Ty)
                t1 = a * S[je-1,je]
                v[je] = ((br * P[je-1,je-1] - a * S[je-1,je-1]) / t1
                         + im * bi * P[je-1,je-1] / t1)
            end
            xmax = max(abs(real(v[je])) + abs(imag(v[je])),
                      abs(real(v[je-1])) + abs(imag(v[je-1])))
            ca = a * v[je-1]
            cb = (br + im * bi) * v[je-1]
            c2a = a * v[je]
            c2b = (br + im * bi) * v[je]
            for jr in 1:je-2
                v[jr] = -ca * S[jr,je-1] + cb * P[jr,je-1] - c2a * S[jr,je] + c2b * P[jr,je]
            end
        end
        dmin = max(ulp * aa * Anorm, ulp * ba * Bnorm, safmin)
        # columnwise triangular solve of (a A - b B) x = 0
        in2x2 = false
        for j in (je - nw):-1:1
            if !in2x2 && j > 1 && S[j,j-1] != 0
                    in2x2 = true
                    continue
            end
            if in2x2
                na = 2
                Bdiag = [P[j,j],P[j+1,j+1]]
            else
                na = 1
                Bdiag = [P[j,j]]
            end
            jx = j + na - 1
            scale, xsum, t1 = _xsolve(a, view(S,j:jx,j:jx), Bdiag, br, bi, view(v,j:jx))
            if scale < one(Ty)
                v[1:je] .= scale * v[1:je]
            end
            xmax = max(scale * xmax, t1)
            v[j:jx] .= xsum
            # w += xⱼ * (a * S[:,j] - b * P[:,j]) with scaling
            if j > 1
                # check whether scaling is needed for xsum
                xscale = one(Ty) / max(one(Ty), xmax)
                t1 = max(t1, aa * Snorms[j] + ba * Pnorms[j])
                if in2x2
                    t1 = max(t1, aa * Snorms[j+1] + ba * Pnorms[j+1])
                end
                t1 = max(t1, aa, ba)
                if t1 > bignum * xscale
                    v[1:je] .= xscale * v[1:je]
                    xmax *= xscale
                end
                # compute contributions from off-diagonals of columns j:jx to sums
                for ja in 1:na
                    ca = a * v[j+ja-1]
                    cb = (br + im * bi) * v[j+ja-1]
                    v[1:j-1] .= v[1:j-1] - ca * S[1:j-1, j+ja-1] + cb * P[1:j-1, j+ja-1]
                end
            end
            in2x2 = false
        end # j loop
        # copy eigvec, backtransforming if needed
        idx -= nw
        if size(Z,1) > 0
            mul!(view(vectors, :, idx), Z, v)
            iend = n
        else
            vectors[:, idx] .= v
            iend = je
        end
        # scale eigvec
        for j in 1:iend
            xmax = max(xmax, abs1(vectors[j,idx]))
        end
        if xmax > safmin
            xscale = one(Ty) / xmax
            vectors[1:iend,idx] .= xscale * vectors[1:iend,idx]
        end
        if inpair
            vectors[:,idx+1] .= conj.(vectors[:,idx])
        end
    end # je loop
    return vectors
end

# left eigenvectors of real generalized Schur
function _gleigvecs(S::StridedMatrix{Ty}, P::StridedMatrix{Ty},
                    Q::StridedMatrix{Ty}=Matrix{Ty}(undef,0,0)
) where {Ty <: Real}
    n = size(S,1)
    ulp = eps(Ty)
    safmin = safemin(Ty)
    smallnum = safmin * (n / ulp)
    bigx = one(Ty) / smallnum
    bignum = one(Ty) / (n * safmin)
    safety = Ty(100)

    vectors = Matrix{Complex{Ty}}(undef,n,n)
    v = zeros(Complex{Ty},n)
    # v is WORK(2n+1:4n)

    # We use the 1-norms of the strictly upper part of S, P columns
    # to avoid overflow
    Anorm = abs(S[1,1])
    Bnorm = abs(P[1,1])
    Snorms = zeros(Ty,n)
    Pnorms = zeros(Ty,n)
    # Snorms and Pnorms are WORK[1:n] and WORK[n+1:2n]
    @inbounds for j=2:n
        for i=1:j-1
            Snorms[j] += abs(S[i,j])
            Pnorms[j] += abs(P[i,j])
        end
        Anorm = max(Anorm,Snorms[j] + abs(S[j,j]))
        Bnorm = max(Bnorm,Pnorms[j] + abs(P[j,j]))
    end
    ascale = one(Ty) / max(Anorm, safmin)
    bscale = one(Ty) / max(Bnorm, safmin)
    idx = 0
    inpair = false
    for je=1:n
        if inpair
            inpair = false
            continue
        end
        nw = 1
        if je < n
            if S[je+1,je] != 0
                inpair = true
                nw = 2
            end
        end
        if !inpair
            if abs(S[je,je]) <= safmin && abs(P[je,je]) <= safmin
                # singular pencil; return unit eigenvector
                idx += 1
                vectors[:,idx] .= zero(T)
                vectors[idx,idx] = one(T)
                continue
            end

            # real eigenvalue
            # compute coeffs a,b in (a A - b B) x = 0
            t = 1 / max(abs(S[je,je]) * ascale,
                       abs(P[je,je]) * bscale, safmin)
            sαr = (t * S[je,je]) * ascale
            sβ = (t * P[je,je]) * bscale
            a = sβ * ascale
            br = sαr * bscale
            bi = zero(Ty)

            # scale to avoid underflow

            lsa = abs(sβ) >= safmin && abs(a) < smallnum
            lsb = abs(sαr) >= safmin && abs(br) < smallnum
            s = one(Ty)
            if lsa
                s = (smallnum / abs(sβ)) * min(Anorm, bigx)
            end
            if lsb
                s = max(s, (smallnum / abs(sαr)) * min(Bnorm, bigx))
            end
            if lsa || lsb
                s = min(s, 1 / (safmin * max( one(Ty), abs(a),
                                            abs(br))))
                if lsa
                    a = ascale * (s * sβ)
                else
                    a *= s
                end
                if lsb
                    br = bscale * (s * sαr)
                else
                    br *= s
                end
            end
            aa = abs(a)
            ba = abs(br)
            v .= zero(Complex{Ty})
            v[je] = one(Ty)
            xmax = one(Ty)
        else
            # complex eigenvalue
            scale1, scale2, wr1, wr2, wi = _ggs_2x2(view(S,je:je+1,je:je+1),
                                                   view(P,je:je+1,je:je+1),safmin*safety)
            a = scale1
            br = wr1
            bi = -wi
            if bi == 0
                throw(ErrorException("algorithm error: real eigval encountered" *
                                     lazy"when complex expected at $je"))
            end
            # scale to avoid over/underflow
            aa = abs(a)
            ba = abs(br) + abs(bi)
            scale = one(Ty)
            if aa * ulp < safmin && aa >= safmin
                scale = (safmin / ulp) / aa
            end
            if ba * ulp < safmin && ba >= safmin
                scale = max(scale, (safmin / ulp) / ba)
            end
            if safmin * aa > ascale
                scale = ascale / (safmin * aa)
            end
            if safmin * ba > bscale
                scale = min(scale, bscale / (safmin * ba))
            end
            if scale != one(Ty)
                a *= scale
                aa = abs(a)
                br *= scale
                bi *= scale
                ba = abs(br) + abs(bi)
            end
            # first two components of eigenvector
            v .= zero(Ty)
            t1 = a * S[je+1,je]
            t2r = a * S[je,je] - br * P[je,je]
            t2i = -bi * P[je,je]
            if abs(t1) > abs(t2r) + abs(t2i)
                v[je] = one(Ty)
                v[je+1] = - (t2r + im * t2i)/ t1
            else
                v[je+1] = one(Ty)
                t1 = a * S[je,je+1]
                v[je] = ((br * P[je+1,je+1] - a * S[je+1,je+1]) / t1
                         + im * bi * P[je+1,je+1] / t1)
            end
            xmax = max(abs(real(v[je])) + abs(imag(v[je])),
                      abs(real(v[je+1])) + abs(imag(v[je+1])))
        end
        dmin = max(ulp * aa * Anorm, ulp * ba * Bnorm, safmin)
        # columnwise triangular solve of (a A - b B) x = 0
        in2x2 = false
        for j in (je + nw):n
            if in2x2
                in2x2 = false
                continue
            end
            if j < n && S[j+1,j] != 0
                in2x2 = true
                na = 2
                Bdiag = [P[j,j],P[j+1,j+1]]
            else
                na = 1
                Bdiag = [P[j,j]]
            end
            # check whether scaling is needed for dot products
            xscale = one(Ty) / max(one(Ty), xmax)
            t1 = max(Snorms[j], Pnorms[j], aa * Snorms[j] + ba * Snorms[j])
            if in2x2
                t1 = max(t1, Snorms[j+1], Pnorms[j+1], aa * Snorms[j+1] + ba * Snorms[j+1])
            end
            if t1 > bignum * xscale
                v[je:j-1] .= v[je:j-1] * xscale
                xmax *= xscale
            end
            # compute dot products
            sum_S = zeros(Complex{Ty},na)
            sum_P = zeros(Complex{Ty},na)
            jx = j + na - 1
            for jr in je:j-1
                sum_S += S[jr,j:jx] .* v[jr]
                sum_P += P[jr,j:jx] .* v[jr]
            end
            spsum = -a * sum_S + (br + im * bi) * sum_P
            scale, y, t1 = _xsolve(a, view(S',j:jx,j:jx), Bdiag, br, bi, spsum)
            v[j:jx] .= y
            if scale < one(Ty)
                v[je:j-1] .= scale * v[je:j-1]
                xmax *= scale
            end
            xmax = max(xmax, t1)
        end # j loop
        # copy eigvec, backtransforming if needed
        idx += 1
        if size(Q,1) > 0
            mul!(view(vectors, :, idx), Q, v)
            ibegin = 1
        else
            vectors[:, idx] .= v
            ibegin = je
        end
        # scale eigvec
        xmax = zero(Ty)
        for j in ibegin:n
            xmax = max(xmax, abs1(vectors[j,idx]))
        end
        if xmax > safmin
            xscale = one(Ty) / xmax
            vectors[ibegin:n,idx] .= xscale * vectors[ibegin:n,idx]
        end
        if inpair
            vectors[:,idx+1] .= conj.(vectors[:,idx])
        end
        idx += nw - 1
    end # je loop
    return vectors
end
