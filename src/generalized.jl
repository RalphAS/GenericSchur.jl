#using LinearAlgebra
#using LinearAlgebra: checksquare, givensAlgorithm, Givens
#using GenericSchur: safemin, abs1, _scale!

# functions for generalized problems

# Developer notes:
# We don't currently implement GeneralizedHessenberg types so anyone else
# using internal functions is doomed to confusion.

using LinearAlgebra: givensAlgorithm

# decomposition is A = Q S Z', B = Q  Tmat Z'

# ggschur! is similar to LAPACK zgges:
# Q is Vsl, Z is Vsr
# Tmat overwrites B
# S overwrites A

function ggschur!(A::StridedMatrix{T}, B::StridedMatrix{T};
                  wantQ::Bool=true, wantZ::Bool=true,
                  scale::Bool=true,
                  kwargs...) where T <: Complex

    n = checksquare(A)
    nb = checksquare(B)
    if nb != n
        throw(ArgumentError("matrices must have the same sizes"))
    end

    if scale
        scaleA, cscale, anrm = _scale!(A)
        scaleB, cscaleb, bnrm = _scale!(B)
    else
        scaleA = false
        scaleB = false
    end
    # maybe balance here

    bqr = qr!(B)
    # apply bqr.Q to A
    lmul!(bqr.Q', A)
    if wantQ
        Q = Matrix(bqr.Q)
    else
        Q = Matrix{T}(undef, 0, 0)
    end
    if wantZ
        Z = Matrix{T}(I, n, n)
    else
        Z =  Matrix{T}(undef, 0, 0)
    end

    # materializing R may waste memory; can we rely on storage in modified B?
    A, B, Q, Z = _hessenberg!(A, bqr.R, Q, Z)
    α, β, S, Tmat, Q, Z = _gqz!(A, B, Q, Z, true)

    # TODO if balancing, unbalance Q, Z

    if scaleA
        safescale!(S, cscale, anrm)
        # checkme: is this always correct?
        α = diag(S)
    end
    if scaleB
        safescale!(Tmat, cscaleb, bnrm)
        # checkme: is this always correct?
        β = diag(Tmat)
    end

    return GeneralizedSchur(S, Tmat, α, β, Q, Z)
end

# B temporarily loses triangularity, so is not specially typed.
# compare to zgghrd
function _hessenberg!(A::StridedMatrix{T}, B::StridedMatrix{T}, Q, Z;
                      ilo=1, ihi=size(A,1)
                      ) where T
    n = checksquare(A)
    wantQ = !isempty(Q)
    wantZ = !isempty(Z)

    triu!(B)

    for jc = ilo:ihi-2
        for jr = ihi:-1:jc+2
            # rotate rows jr-1,jr to null A[jr,jc]
            Gq,r = givens(A,jr-1,jr,jc)
            lmul!(Gq, A)
            lmul!(Gq, B)
            if wantQ
                rmul!(Q, Gq')
            end
            # rotate cols jr,jr-1 to null B[jr,jr-1]
            Gz,r = givens(B',jr,jr-1,jr)
            rmul!(A,Gz')
            rmul!(B,Gz')
            if wantZ
                rmul!(Z, Gz')
            end
        end
    end

    return triu!(A,-1), triu!(B), Q, Z
end

# translated from zhgeqz
# single-shift QZ algo
# H is Hessenberg
# B is upper-triangular
function _gqz!(H::StridedMatrix{T}, B::StridedMatrix{T}, Q, Z, wantSchur;
               debug = false,
               ilo=1, ihi=size(H,1), maxiter=100*(ihi-ilo+1)
               ) where {T <: Complex}
    n = checksquare(H)
    wantQ = !isempty(Q)
    wantZ = !isempty(Z)

    if ilo > 1 || ihi < n
        @warn "almost surely not working for nontrivial ilo,ihi"
    end
    α = Vector{T}(undef, n)
    β = Vector{T}(undef, n)

    RT = real(T)
    ulp = eps(RT)
    safmin = safemin(RT)
    anorm = norm(view(H,ilo:ihi,ilo:ihi),2) # Frobenius
    bnorm = norm(view(B,ilo:ihi,ilo:ihi),2) # Frobenius
    atol = max(safmin, ulp * anorm)
    btol = max(safmin, ulp * bnorm)
    ascale = one(RT) / max(safmin, anorm)
    bscale = one(RT) / max(safmin, bnorm)
    half = 1 / RT(2)

    # TODO: if ihi < n, deal with ihi+1:n

    ifirst = ilo
    ilast = ihi

    if wantSchur
        ifirstm, ilastm = 1,n
    else
        ifirstm, ilastm = ilo,ihi
    end

    shiftcount = 0
    eshift = zero(T)

    alldone = false
    for it=1:maxiter
        # warning: _istart, ifirst, ilast are as in LAPACK, not GLA
        # note zlartg(f,g,c,s,r) -> (c,s,r) = givensAlgorithm(f,g)

        # Split H if possible
        # Use 2 tests:
        # 1: H[j,j-1] == 0 || j=ilo
        # 2: B[j,j] = 0

        trivsplit = false
        if ilast == ilo
            trivsplit = true
        elseif abs1(H[ilast,ilast-1]) < atol
            H[ilast, ilast-1] = 0
            trivsplit = true
        end
        # For now, we follow LAPACK too closely.
        # br is a branch indicator corresponding to goto targets
        br = trivsplit ? :br60 : :br_none
        if !trivsplit # 60
            if abs(B[ilast,ilast]) <= btol
                B[ilast,ilast] = 0
                br = :br50
            else
                # general case
                for j=ilast-1:-1:ilo
                    # Test 1
                    ilazro = false
                    if j==ilo
                        ilazro = true
                    else
                        if abs1(H[j,j-1]) <= atol
                            H[j,j-1] = 0
                            ilazro = true
                        end
                    end
                    # Test 2
                    if abs(B[j,j]) < btol
                        B[j,j] = 0
                        # Test 1a: check for 2 consec. small subdiags in H
                        ilazr2 = !ilazro &&
                            (abs1(H[j,j-1]) * (ascale * abs1(H[j+1,j]))
                             <= abs1(H[j,j]) * (ascale * atol))
                        # if both tests pass, split 1x1 block off top
                        # if remaining leading diagonal elt vanishes, iterate
                        if ilazro || ilazr2
                            for jch=j:ilast-1
                                G,r = givens(H[jch,jch],H[jch+1,jch],jch,jch+1)
                                H[jch,jch] = r
                                H[jch+1,jch] = 0
                                lmul!(G,view(H,jch:ihi,jch+1:ilastm))
                                lmul!(G,view(B,jch:ihi,jch+1:ilastm))
                                if wantQ
                                    rmul!(Q, G')
                                end
                                if ilazr2
                                    H[jch,jch-1] *= c
                                end
                                ilazr2 = false
                                if abs1(B[jch+1, jch+1]) > btol
                                    if jch+1 >= ilast
                                        br = :br60
                                        break
                                    else
                                        ifirst = jch+1
                                        br = :br70
                                        break
                                    end
                                end
                                B[jch+1, jch+1] = 0
                            end # jch loop
                            br = :br50
                            break
                        else
                            # only test 2 passed: chase 0 to B[ilast,ilast]
                            # then process as above
                            for jch=j:ilast-1
                                G,r = givens(B[jch,jch+1],B[jch+1,jch+1],
                                             jch,jch+1)
                                B[jch,jch+1] = r
                                B[jch+1,jch+1] = 0
                                if jch < ilastm-1
                                    lmul!(G,view(B,:,jch+2:ilastm))
                                end
                                lmul!(G,view(H,:,jch-1:ilastm))
                                if wantQ
                                    rmul!(Q,G)
                                end
                                G,r = givens(H[jch+1,jch],H[jch+1,jch-1],
                                             jch,jch-1)
                                H[jch+1,jch-1] = 0
                                # checkme G or G'?
                                rmul!(view(H,ifirstm:jch,:),G')
                                rmul!(view(B,ifrstm:jch+1,:),G')
                                if wantZ
                                    rmul!(Z,G')
                                end
                            end # jch loop
                        end # if ilazro || ilazr2
                    elseif ilazro # not tiny B[j,j]
                        ifirst = j
                        br = :br70
                        break
                    end # if B[j,j] tiny
                    # neither test passed; try next j
                end # j loop (40)
                if br == :br_none
                    error("dev error: drop-through assumed impossible")
                end
            end # if B[ilast,ilast] tiny
            if br == :br50
                # B[ilast,ilast] = 0; clear H[ilast,ilast-1] to split
                G,r = givens(H[ilast,ilast],H[ilast,ilast-1],ilast,ilast-1)
                H[ilast,ilast] = r
                H[ilast,ilast-1] = 0
                rmul!(view(H,ifrstm:ilast-1,:),G')
                rmul!(view(B,ifrstm:ilast-1,:),G')
                if wantZ
                    rmul!(Z,G')
                end
                br = :br60
            end
        end # if !trivsplit
        if br == :br60
            # H[ilast, ilast-1] == 0: standardize B, set α,β
            absb = abs(B[ilast,ilast])
            if absb > safmin
                signbc = conj(B[ilast,ilast] / absb)
                B[ilast,ilast] = absb
                if wantSchur
                    B[ifirstm:ilast-1,ilast] .*= signbc
                    H[ifirstm:ilast,ilast] .*= signbc
                else
                    H[ilast,ilast] *= signbc
                end
                if wantZ
                    Z[:,ilast] .*= signbc
                end
            else
                B[ilast,ilast] = 0
            end
            α[ilast] = H[ilast,ilast]
            β[ilast] = B[ilast,ilast]
            ilast -= 1
            if ilast < ilo
                br = :br190
                break
            end
            shiftcount = 0
            eshift = zero(T)
            if !wantSchur
                ilastm = ilast
                if ifirstm > ilast
                    ifirstm = ilo
                end
            end
            continue # iteration loop
        end # br == :br60

        @assert br == :br70

        # QZ step
        # This iteration involves block ifirst:ilast, assumed nonempty
        if !wantSchur
            ifirstm = ifirst
        end

        # Compute shift
        shiftcount += 1
        # At this point ifirst < ilast and diagonal elts of B in that block
        # are all nontrivial.
        if shiftcount % 10 !=  0
            # Wilkinson shift, i.e. eigenvalue of last 2x2 block of
            # A B⁻¹ nearest to last elt

            # factor B as U D where U has unit diagonal, compute A D⁻¹ U⁻¹
            b11 = (bscale * B[ilast-1,ilast-1])
            b22 = (bscale * B[ilast,ilast])
            u12 = (bscale * B[ilast-1,ilast]) / b22
            ad11 = (ascale * H[ilast-1,ilast-1]) / b11
            ad21 = (ascale * H[ilast,ilast-1]) / b11
            ad12 = (ascale * H[ilast-1,ilast]) / b22
            ad22 = (ascale * H[ilast,ilast]) / b22
            abi22 = ad22 - u12 * ad21
            t1 = half * (ad11 + abi22)
            rtdisc = sqrt(t1^2 + ad12 * ad21 - ad11 * ad22)
            t2 = real(t1 - abi22) * real(rtdisc) + imag(t1-abi22) * imag(rtdisc)
            shift =  (t2 <= 0) ? (t1 + rtdisc) : (t1 - rtdisc)
        else
            # exceptional shift
            # "Chosen for no particularly good reason" (LAPACK)
            eshift += (ascale * H[ilast, ilast-1]) / (bscale * B[ilast-1, ilast-1])
            shift = eshift
        end

        # check for two consecutive small subdiagonals
        local f1
        gotit = false
        for j=ilast-1:-1:ifirst+1
            _istart = j
            f1 = ascale * H[j,j] - shift * (bscale * B[j,j])
            t1 = abs1(f1)
            t2 = ascale * abs1(H[j+1,j])
            tr = max(t1,t2)
            if tr < one(RT) && tr != zero(RT)
                t1 /= tr
                t2 /= tr
            end
            if abs1(H[j,j-1]) * t2 <= t1 * atol
                gotit = true
                break
            end
        end
        if !gotit
            _istart = ifirst
            f1 = (ascale * H[ifirst, ifirst]
                  - shift * (bscale * B[ifirst, ifirst]))
        end

        # qz sweep
        g2 = ascale*H[_istart+1, _istart]
        c,s,t3 = givensAlgorithm(f1, g2)
        for j=_istart:ilast-1
            if j > _istart
                    c,s,r = givensAlgorithm(H[j,j-1], H[j+1,j-1])
                    H[j,j-1] = r
                    H[j+1,j-1] = 0
            end
            G = Givens(j,j+1,T(c),s)
            lmul!(G, view(H,:,j:ilastm))
            lmul!(G, view(B,:,j:ilastm))
            if wantQ
                rmul!(Q, G')
            end
            c,s,r = givensAlgorithm(B[j+1,j+1], B[j+1,j])
            B[j+1,j+1] = r
            B[j+1,j] = 0
            G = Givens(j,j+1,T(c),s)
            rmul!(view(H,ifirstm:min(j+2,ilast),:),G)
            rmul!(view(B,ifirstm:j,:),G)
            if wantZ
                rmul!(Z, G)
            end
        end
        if it >= maxiter
            @warn "convergence failure; factorization incomplete"
            break
        end
    end # iteration loop

    # TODO if ilo > 1, deal with 1:ilo-1

    return α, β, H, B, Q, Z
end

function eigvecs(S::GeneralizedSchur, left=false)
    left && throw(ArgumentError("not implemented"))
    return _geigvecs(S.S, S.T, S.Z)
end


# right eigenvectors
# translated from LAPACK::ztgevc
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
function _geigvecs(S::StridedMatrix{T}, P::StridedMatrix{T},
                    Z::StridedMatrix{T}=Matrix{T}(undef,0,0)
                    ) where {T <: Complex}
    n = size(S,1)
    RT = real(T)
    ulp = eps(RT)
    safmin = safemin(RT)
    smallnum = safmin * (n / ulp)
    big = one(RT) / smallnum
    bignum = one(RT) / (n * safmin)

    vectors = Matrix{T}(undef,n,n)
    v = zeros(T,n)
    if size(Z,1) > 0
        v2 = zeros(T,n)
    end

    # We use the 1-norms of the strictly upper part of S columns
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
                s = (smallnum / abs(sβ)) * min(anorm, big)
            end
            if lsb
                s = max(s, (smallnum / abs1(sα)) * min(bnorm, big))
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
            v[ki] = one(T)
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
