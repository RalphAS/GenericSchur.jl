# This file is part of GenericSchur.jl, released under the MIT "Expat" license
# Portions derived from LAPACK, see below.

# Real Schur decomposition (QZ algorithm) for generalized eigen-problems

# decomposition is A = Q S Z', B = Q T Z'

function schur!(A::StridedMatrix{T}, B::StridedMatrix{T}; kwargs...) where {T<:Real}
    ggschur!(A, B; kwargs...)
end

# similar to LAPACK dgges.
# TODO: consolidate w/ complex version if possible.
function ggschur!(A::StridedMatrix{T}, B::StridedMatrix{T};
                  wantQ::Bool=true, wantZ::Bool=true,
                  scale::Bool=true,
                  kwargs...) where T <: Real

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
        safescale!(α, cscale, anrm)
    end
    if scaleB
        safescale!(Tmat, cscaleb, bnrm)
        safescale!(β, cscaleb, bnrm)
    end

    return GeneralizedSchur(S, Tmat, α, β, Q, Z)
end

const _RG2X2_SAFETY = 100

# translated from dhgeqz (LAPACK)
function _gqz!(H::StridedMatrix{T}, B::StridedMatrix{T}, Q, Z, wantSchur;
               debug = false,
               ilo=1, ihi=size(H,1), maxiter=100*(ihi-ilo+1)
               ) where {T <: Real}
    n = checksquare(H)
    wantQ = !isempty(Q)
    wantZ = !isempty(Z)

    α = Vector{complex(T)}(undef, n)
    β = Vector{T}(undef, n)

    ulp = eps(T)
    safmin = safemin(T)
    safmax = one(T) / safmin
    safety = T(_RG2X2_SAFETY)
    anorm = norm(view(H,ilo:ihi,ilo:ihi),2) # Frobenius
    bnorm = norm(view(B,ilo:ihi,ilo:ihi),2) # Frobenius
    atol = max(safmin, ulp * anorm)
    btol = max(safmin, ulp * bnorm)
    ascale = one(T) / max(safmin, anorm)
    bscale = one(T) / max(safmin, bnorm)
    half = 1 / T(2)

    # set trivial trailing eigvals
    for j in ihi+1:n
        if B[j,j] < 0
            if wantSchur
                H[1:j,j] .= -H[1:j,j]
                B[1:j,j] .= -B[1:j,j]
            else
                H[j,j] = -H[j,j]
                B[j,j] = -B[j,j]
            end
            if wantZ
                Z[1:n,j] .= -Z[1:n,j]
            end
        end
        α[j] = H[j,j]
        β[j] = B[j,j]
    end

    ifirst = ilo
    ilast = ihi

    if wantSchur
        ifirstm, ilastm = 1,n
    else
        ifirstm, ilastm = ilo,ihi
    end

    shiftcount = 0
    eshift = zero(T)
    v = zeros(T,3) # used for reflectors below

    for it=1:maxiter
        @mydebug println("iteration $it ilast=$ilast")

        # Split H if possible
        # Use 2 tests:
        # 1: H[j,j-1] == 0 || j=ilo
        # 2: B[j,j] = 0

        trivsplit = false
        if ilast == ilo
            trivsplit = true
        elseif (abs(H[ilast,ilast-1]) <
            max(safmin, ulp * (abs(H[ilast,ilast]) + abs(H[ilast-1,ilast-1]))))
            H[ilast, ilast-1] = 0
            trivsplit = true
        end
        if trivsplit
            @mydebug println("trivial split at $ilast")
        end
        # For now, we follow LAPACK too closely.
        # br is a branch indicator corresponding to goto targets
        br = trivsplit ? :br_deflate_ilast : :br_none
        if !trivsplit
            if abs(B[ilast,ilast]) <= btol
                B[ilast,ilast] = 0
                br = :br_B_ilast_sing
                @mydebug println("split on singular entry in B at $ilast")
            else
                # general case
                for j=ilast-1:-1:ilo
                    # Test 1
                    have_H_0sd = false
                    if j==ilo
                        have_H_0sd = true
                    else
                        if abs(H[j,j-1]) <= atol
                            H[j,j-1] = 0
                            have_H_0sd = true
                        end
                    end
                    # Test 2
                    if abs(B[j,j]) < btol
                        B[j,j] = 0
                        # Test 1a: check for 2 consec. small subdiags in H
                        have2_H_0sd = false
                        if !have_H_0sd
                            t = abs(H[j,j-1])
                            t2 = abs(H[j,j])
                            tr = max(t,t2)
                            if tr < 1 && tr != 0
                                t = t / tr
                                t2 = t2 / tr
                            end
                            have2_H_0sd = (t * (ascale * abs(H[j+1,j]))
                                      <= t2 * (ascale * atol))
                        end
                        # if both tests pass, split 1x1 block off top
                        # if remaining leading diagonal elt vanishes, iterate
                        if have_H_0sd || have2_H_0sd
                            @mydebug println("split on singular entry in B at $j")
                            for jch=j:ilast-1
                                # G,r = givens(H[jch,jch],H[jch+1,jch],jch,jch+1)
                                c,s,r = givensAlgorithm(H[jch,jch],H[jch+1,jch])
                                G = Givens(jch,jch+1,c,s)
                                H[jch,jch] = r
                                H[jch+1,jch] = 0
                                lmul!(G,view(H,:,jch+1:ilastm))
                                lmul!(G,view(B,:,jch+1:ilastm))
                                if wantQ
                                    rmul!(Q, G')
                                end
                                if have2_H_0sd
                                    H[jch,jch-1] *= c
                                end
                                have2_H_0sd = false
                                if abs(B[jch+1, jch+1]) > btol
                                    if jch+1 >= ilast
                                        br = :br_deflate_ilast
                                        break
                                    else
                                        ifirst = jch+1
                                        br = :br_QZ
                                        break
                                    end
                                end
                                @mydebug println("  another at $(jch+1)")
                                B[jch+1, jch+1] = 0
                            end # jch loop
                            br = :br_B_ilast_sing
                            break
                        else
                            # only test 2 passed: chase 0 to B[ilast,ilast]
                            # then process as above
                            @mydebug println("chase singular entry in B at $j")
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
                                    rmul!(Q,G')
                                end
                                G,r = givens(H[jch+1,jch],H[jch+1,jch-1],
                                            jch,jch-1)
                                H[jch+1,jch] = r
                                H[jch+1,jch-1] = 0
                                rmul!(view(H,ifirstm:jch,:),G')
                                rmul!(view(B,ifirstm:jch-1,:),G')
                                if wantZ
                                    rmul!(Z,G')
                                end
                            end # jch loop
                            br = :br_B_ilast_sing
                            break
                        end # if have_H_0sd || have2_H_0sd
                    elseif have_H_0sd # not tiny B[j,j]
                        ifirst = j
                        @mydebug (ifirst > ilo) && println("nonsingular split at $ifirst")
                        br = :br_QZ
                        break
                    end # if B[j,j] tiny
                    # neither test passed; try next j
                end # j loop (40)
                if br == :br_none
                    error("algorithmic failure: splitting drop-through assumed impossible")
                end
            end # if B[ilast,ilast] tiny
            if br == :br_B_ilast_sing
                # B[ilast,ilast] = 0; clear H[ilast,ilast-1] to split off 1x1
                G,r = givens(H[ilast,ilast],H[ilast,ilast-1],ilast,ilast-1)
                H[ilast,ilast] = r
                H[ilast,ilast-1] = 0
                rmul!(view(H,ifirstm:ilast-1,:),G')
                rmul!(view(B,ifirstm:ilast-1,:),G')
                if wantZ
                    rmul!(Z,G')
                end
                br = :br_deflate_ilast
            end
        end # if !trivsplit
        if br == :br_deflate_ilast
            # found or made zero subdiag H[ilast, ilast-1]:
            @mydebug println("deflating $ilast")
            # standardize B, set α,β
            if B[ilast,ilast] < 0
                if wantSchur
                    H[ifirstm:ilast, ilast] .= -H[ifirstm:ilast, ilast]
                    B[ifirstm:ilast, ilast] .= -B[ifirstm:ilast, ilast]
                else
                    H[ilast, ilast] = -H[ilast, ilast]
                    B[ilast, ilast] = -B[ilast, ilast]
                end
                if wantZ
                    Z[:, ilast] .= -Z[:,ilast]
                end
            end

            α[ilast] = H[ilast,ilast]
            β[ilast] = B[ilast,ilast]

            # advance to next block or exit if done with core blocks
            ilast -= 1
            if ilast < ilo
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
        end # br == :br_deflate_ilast branch

        @assert br == :br_QZ

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

            s1,s2,wr,wr2,wi = _ggs_2x2(view(H, ilast-1:ilast, ilast-1:ilast),
                                      view(B, ilast-1:ilast, ilast-1:ilast),
                                      safmin * safety)
            if (abs((wr/s1) * B[ilast,ilast] - H[ilast, ilast])
                > abs((wr2 / s2) * B[ilast, ilast] - H[ilast, ilast]))
                wr, wr2 = wr2, wr
                s1, s2 = s2, s1
            end

        else
            # exceptional shift
            # "Chosen for no particularly good reason" (LAPACK)
            if maxiter * safmin * abs(H[ilast, ilast-1]) < abs(B[ilast-1,ilast-1])
                eshift = H[ilast, ilast-1] / B[ilast-1, ilast-1]
            else
                eshift += 1 / (safmin * maxiter)
            end
            s1 = T(1)
            wr = eshift
            wi = T(0)
        end


      if wi == 0
        # real shift; logic (approx) equiv to zhgeqz
          @mydebug println("QZ $ifirst:$ilast w/ shift $wr"
              * "(exc)"^(shiftcount % 10 == 0))

        # fiddle with shift to avoid overflow
        t = min(ascale, one(T)) * (safmax / 2)
        scale = s1 > t ? t / s1 : one(T)
        t = min(bscale, one(T)) * (safmax / 2)
        if abs(wr) > t
            scale = min(scale, t / abs(wr))
        end
        s1 *= scale
        wr *= scale

        # check for two consecutive small subdiagonals
        local f1
        gotit = false
        for j=ilast-1:-1:ifirst+1
            _istart = j
            t = abs(s1*H[j,j-1])
            t2 = abs(s1*H[j,j] - wr*B[j,j])
            tr = max(t,t2)
            if tr < 1 && tr != 0
                t /= tr
                t2 /= tr
            end
            if abs((ascale * H[j+1, j]) * t) <= ascale * atol * t2
                gotit = true
                break
            end
        end
        if !gotit
            _istart = ifirst
        end

        # implicit single-shift qz sweep
        f1 = s1 * H[_istart, _istart] - wr * B[_istart, _istart]
        g2 = s1 * H[_istart+1, _istart]
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
      else # shift is complex
          @mydebug println("QZ $ifirst:$ilast w/ shift $wr + $wi im")
          # Francis double shift
          # dhgeqz notes that this should work with real shifts if block is at least 3x3,
          # but might fail in 2x2 case
          if ilast == ifirst+1
              # Special case: 2x2 block w/ complex eigenvectors

              # step 1: standardize (diagonalize B block)
              # Note ordering gotcha: fn returns ssmin,ssmax,...
              # We want largest s.v. in first slot.
              b22,b11,sr,cr,sl,cl = _gsvd_2x2(view(B,ifirst:ilast,ifirst:ilast))
              if b11 < 0
                  cr = -cr
                  sr = -sr
                  b11 = -b11
                  b22 = -b22
              end
              Gl = Givens(ifirst,ilast,cl,sl)
              Gr = Givens(ifirst,ilast,cr,sr)
              lmul!(Gl, view(H,:,ifirst:ilastm))
              rmul!(view(H,ifirstm:ilast,:), Gr')
              if ilast < ilastm
                  lmul!(Gl, view(B,:,ilast+1:ilastm))
              end
              if ifirstm < ifirst
                  rmul!(view(B,ifirstm:ifirst-1,:), Gr')
              end
              # these might conceivably be correct
              if wantQ
                  rmul!(Q, Gl')
              end
              if wantZ
                  rmul!(Z, Gr')
              end

              B[ilast-1, ilast-1] = b11
              B[ilast-1, ilast] = 0
              B[ilast, ilast-1] = 0
              B[ilast, ilast] = b22
              if b22 < 0
                  H[ifirstm:ilast, ilast] .= - H[ifirstm:ilast, ilast]
                  B[ifirstm:ilast, ilast] .= - B[ifirstm:ilast, ilast]
                  if wantZ
                      Z[1:n, ilast] .= -Z[1:n,ilast]
                  end
                  b22 = - b22
              end
              # step 2: compute α, β
              s1, _, wr, _, wi = _ggs_2x2(view(H,ilast-1:ilast, ilast-1:ilast),
                                         view(B,ilast-1:ilast, ilast-1:ilast),
                                         safmin * safety)
              if wi == 0 # perturbation put shift back on real line
                  @mydebug println("backtracking to real shift")
                  @mydebug begin
                  @show eigvals(view(H,ilast-1:ilast, ilast-1:ilast),
                                         view(B,ilast-1:ilast, ilast-1:ilast))
                  end
                  # go back for another single shift
                  continue
              end
              s1inv = 1 / s1
              a11 = H[ilast-1,ilast-1]
              a21 = H[ilast,ilast-1]
              a12 = H[ilast-1,ilast]
              a22 = H[ilast,ilast]

              # complex Givens on right
              # assume some elt of sA-wB > unfl
              c11r = s1 * a11 - wr * b11
              c11i = -wi * b11
              c12 = s1 * a12
              c21 = s1 * a21
              c22r = s1 * a22 - wr * b22
              c22i = -wi * b22
              if abs(c11r) + abs(c11i) + abs(c12) > abs(c21) + abs(c22r) + abs(c22i)
                  t1 = hypot(c12, c11r, c11i)
                  cz = c12 / t1
                  szr = -c11r / t1
                  szi = -c11i / t1
              else
                  cz = hypot(c22r, c22i)
                  if cz <= safmin
                      cz = zero(T)
                      szr = one(T)
                      szi = zero(T)
                  else
                      tr = c22r / cz
                      ti = c22i / cz
                      t1 = hypot(cz, c21)
                      cz /= t1
                      szr = -c21 * tr / t1
                      szi = c21 * ti / t1
                  end
              end
              # Givens on left
              an = abs(a11) + abs(a12) + abs(a21) + abs(a22)
              bn = abs(b11) + abs(b22)
              wabs = abs(wr) + abs(wi)
              if s1 * an > wabs * bn
                  cq = cz * b11
                  sqr = szr * b22
                  sqi = -szi * b22
              else
                  a1r = cz * a11 + szr * a12
                  a1i = szi * a12
                  a2r = cz * a21 + szr * a22
                  a2i = szi * a22
                  cq = hypot(a1r, a1i)
                  if cq <= safmin
                      cq = zero(T)
                      sqr = one(T)
                      sqi = zero(T)
                  else
                      tr = a1r / cq
                      ti = a1i / cq
                      sqr = tr * a2r + ti * a2i
                      sqi = ti * a2r - tr * a2i
                  end
              end
              t1 = hypot(cq, sqr, sqi)
              cq /= t1
              sqr /= t1
              sqi /= t1
              # compute diagonal elts of QBZ
              tr = sqr * szr - sqi * szi
              ti = sqr * szi + sqi * szr
              b1r = cq * cz * b11 + tr * b22
              b1i = ti * b22
              b1a = hypot(b1r, b1i)
              b2r = cq * cz * b22 + tr * b11
              b2i = -ti * b11
              b2a = hypot(b2r, b2i)
              # standardize
              @mydebug println("deflating complex pair")
              β[ilast - 1] = b1a
              β[ilast] = b2a
              α[ilast - 1] = (complex(wr, wi) * b1a) * s1inv
              α[ilast] = (complex(wr, -wi) * b2a) * s1inv
              ilast = ifirst - 1
              if ilast < ilo # done with core block
                  break
              end
              # reset counters and go on to next block
              shiftcount = 0
              eshift = zero(T)
              if !wantSchur
                  ilastm = ilast
                  if ifirstm > ilast
                      ifirstm = ilo
                  end
              end
          else
              # usual case: 3x3 or larger
              # use Francis implicit double shift

              # eigval eqn is ω^2 - c ω + d = 0
              # compute first column of (AB⁻¹)² - c AB⁻¹ + d
              # assuming block is at least 3x3
              ad11 = ( ascale*H[ ilast-1, ilast-1 ] ) /( bscale*B[ ilast-1, ilast-1 ] )
              ad21 = ( ascale*H[ ilast, ilast-1 ] ) /( bscale*B[ ilast-1, ilast-1 ] )
              ad12 = ( ascale*H[ ilast-1, ilast ] ) /( bscale*B[ ilast, ilast ] )
              ad22 = ( ascale*H[ ilast, ilast ] ) /( bscale*B[ ilast, ilast ] )
              u12 = B[ ilast-1, ilast ] / B[ ilast, ilast ]
              ad11l = ( ascale*H[ ifirst, ifirst ] ) /( bscale*B[ ifirst, ifirst ] )
              ad21l = ( ascale*H[ ifirst+1, ifirst ] ) /( bscale*B[ ifirst, ifirst ] )
              ad12l = ( ascale*H[ ifirst, ifirst+1 ] ) /( bscale*B[ ifirst+1, ifirst+1 ] )
              ad22l = ( ascale*H[ ifirst+1, ifirst+1 ] ) /( bscale*B[ ifirst+1, ifirst+1 ] )
              ad32l = ( ascale*H[ ifirst+2, ifirst+1 ] ) /( bscale*B[ ifirst+1, ifirst+1 ] )
              u12l = B[ ifirst, ifirst+1 ] / B[ ifirst+1, ifirst+1 ]

              v[1] = (( ad11-ad11l )*( ad22-ad11l ) - ad12*ad21 +ad21*u12*ad11l
                        + ( ad12l-ad11l*u12l )*ad21l)
              v[2] = (( ( ad22l-ad11l )-ad21l*u12l-( ad11-ad11l )-( ad22-ad11l )
                          +ad21*u12 ))*ad21l
              v[3] = ad32l*ad21l
              _istart = ifirst

              τ = _reflector!(v)
              v[1] = one(T)
              # sweep
              for j in _istart:ilast-2
                  if j > _istart
                      v .= H[j:j+2,j-1]
                      τ = _reflector!(v)
                      H[j,j-1] = v[1]
                      v[1] = one(T)
                      H[j+1:j+2, j-1] .= zero(T)
                  end
                  t2 = τ * v[2]
                  t3 = τ * v[3]
                  for jc in j:ilastm
                      t1 = H[j,jc] + v[2] * H[j+1,jc] + v[3] * H[j+2,jc]
                      H[j,jc] -= t1 * τ
                      H[j+1,jc] -= t1 * t2
                      H[j+2,jc] -= t1 * t3
                      t1 = B[j,jc] + v[2] * B[j+1,jc] + v[3] * B[j+2,jc]
                      B[j,jc] -= t1 * τ
                      B[j+1,jc] -= t1 * t2
                      B[j+2,jc] -= t1 * t3
                  end
                  if wantQ
                      for jr in 1:n
                          t1 = Q[jr,j] + v[2] * Q[jr,j+1] + v[3] * Q[jr,j+2]
                          Q[jr,j] -= t1 * τ
                          Q[jr,j+1] -= t1 * t2
                          Q[jr,j+2] -= t1 * t3
                      end
                  end
                  # zero j-th column of B
                  # swap rows to pivot
                  pivot = false
                  t1 = max(abs(B[j+1,j+1]), abs(B[j+1,j+2]))
                  t2 = max(abs(B[j+2,j+1]), abs(B[j+2,j+2]))
                  if max(t1,t2) < safmin
                      scale = zero(T)
                      u1 = one(T)
                      u2 = zero(T)
                  else
                      if t1 >= t2
                          w11 = B[j+1,j+1]
                          w21 = B[j+2,j+1]
                          w12 = B[j+1,j+2]
                          w22 = B[j+2,j+2]
                          u1 = B[j+1,j]
                          u2 = B[j+2,j]
                      else
                          w21 = B[j+1,j+1]
                          w11 = B[j+2,j+1]
                          w22 = B[j+1,j+2]
                          w12 = B[j+2,j+2]
                          u2 = B[j+1,j]
                          u1 = B[j+2,j]
                      end
                      if abs(w12) > abs(w11)
                          pivot = true
                          w11,w12 = w12,w11
                          w21,w22 = w22,w21
                      end
                      # LU factor
                      t1 = w21 / w11
                      u2 -= t1 * u1
                      w22 -= t1 * w12
                      w21 = zero(T)
                      # compute scale
                      scale = one(T)
                      if abs(w22) < safmin
                          scale = zero(T)
                      else
                          if abs(w22) < abs(u2)
                              scale = abs(w22 / u2)
                          end
                          if abs(w11) < abs(u1)
                              scale = min(scale, abs(w11 / u1))
                          end
                          # solve
                          u2 = (scale * u2) / w22
                          u1 = (scale * u1 - w12 * u2) / w11
                      end
                  end
                  if pivot
                      u1,u2 = u2,u1
                  end
                  # Householder vector
                  t1 = hypot(scale, u1, u2)
                  τ = one(T) + scale / t1
                  vs = -one(T) / (scale + t1)
                  v[1] = one(T)
                  v[2] = vs * u1
                  v[3] = vs * u2
                  t2 = τ * v[2]
                  t3 = τ * v[3]
                  for jr in ifirstm:min(j+3, ilast)
                      t1 = H[jr,j] + v[2] * H[jr,j+1] + v[3] * H[jr,j+2]
                      H[jr,j] -= t1 * τ
                      H[jr,j+1] -= t2 * t1
                      H[jr,j+2] -= t3 * t1
                  end
                  for jr in ifirstm:j+2
                      t1 = B[jr,j] + v[2] * B[jr,j+1] + v[3] * B[jr,j+2]
                      B[jr,j] -= t1 * τ
                      B[jr,j+1] -= t2 * t1
                      B[jr,j+2] -= t3 * t1
                  end
                  if wantZ
                      for jr in 1:n
                          t1 = Z[jr,j] + v[2] * Z[jr,j+1] + v[3] * Z[jr,j+2]
                          Z[jr,j] -= t1 * τ
                          Z[jr,j+1] -= t2 * t1
                          Z[jr,j+2] -= t3 * t1
                      end
                  end
                  B[j+1:j+2, j] .= zero(T)
              end # sweep

              # use Givens rotations for last elements
              # rotations from left
              j = ilast - 1
              c,s,r = givensAlgorithm(H[j,j-1], H[j+1,j-1])
              H[j,j-1] = r
              H[j+1,j-1] = zero(T)
              for jc in j:ilastm
                  t1 = c * H[j,jc] + s * H[j+1,jc]
                  H[j+1,jc] = -s * H[j,jc] + c * H[j+1,jc]
                  H[j,jc] = t1
                  t2 = c * B[j,jc] + s * B[j+1,jc]
                  B[j+1,jc] = -s * B[j,jc] + c * B[j+1,jc]
                  B[j,jc] = t2
              end
              if wantQ
                  for jr in 1:n
                      t1 = c * Q[jr,j] + s * Q[jr,j+1]
                      Q[jr,j+1] = -s * Q[jr,j] + c * Q[jr,j+1]
                      Q[jr,j] = t1
                  end
              end
              # rotations from left
              c,s,r = givensAlgorithm(B[j+1,j+1], B[j+1,j])
              B[j+1,j+1] = r
              B[j+1,j] = zero(T)
              for jr in ifirstm:ilast
                  t1 = c * H[jr,j+1] + s * H[jr,j]
                  H[jr,j] = -s * H[jr,j+1] + c * H[jr,j]
                  H[jr,j+1] = t1
              end
              for jr in ifirstm:ilast-1
                  t1 = c * B[jr,j+1] + s * B[jr,j]
                  B[jr,j] = -s * B[jr,j+1] + c * B[jr,j]
                  B[jr,j+1] = t1
              end
              if wantZ
                  for jr in 1:n
                      t1 = c * Z[jr,j+1] + s * Z[jr,j]
                      Z[jr,j] = -s * Z[jr,j+1] + c * Z[jr,j]
                      Z[jr,j+1] = t1
                  end
              end

          end # 2x2 vs larger block branches
      end # single/double shift branches

        if it >= maxiter
            throw(UnconvergedException("iteration limit $maxiter reached"))
        end
    end # iteration loop

    # set eigenvalues for trivial leading part
    for j in 1:ilo-1
        if B[j,j] < 0
            if wantSchur
                H[1:j,j] .= -H[1:j,j]
                B[1:j,j] .= -B[1:j,j]
            else
                H[j,j] = -H[j,j]
                B[j,j] = -B[j,j]
            end
            if wantZ
                Z[1:n, j] .= -Z[1:n, j]
            end
        end
        α[j] = H[j,j]
        β[j] = B[j,j]
    end

    return α, β, H, B, Q, Z
end

"""
`scale1, scale2, wr1, wr2, wi = _ggs_2x2(A,B,safemin)`

Compute eigenvalues of real generalized 2x2 problem (A - w B) with scaling to avoid
over-/underflow. Actual eigvals are `(wr1 + im * wi) / scale1` etc.
B must be upper triangular. See LAPACK dlag2 for other constraints on A,B.
"""
function _ggs_2x2(A::AbstractMatrix{T},B,safmin) where {T <: AbstractFloat}
    # translation of LAPACK dlag2
    rtmin = sqrt(safmin)
    rtmax = one(T) / rtmin
    safmax = one(T) / safmin
    fuzzy1 = one(T) + sqrt(sqrt(eps(one(T)))) / 10

    anorm = max(abs(A[1,1]) + abs(A[2,1]), abs(A[1,2]) + abs(A[2,2]), safmin)
    ascale = one(T) / anorm
    a11 = ascale * A[1,1]
    a12 = ascale * A[1,2]
    a21 = ascale * A[2,1]
    a22 = ascale * A[2,2]

    # perturb to avoid singularity if needed
    b11 = B[1,1]
    b12 = B[1,2]
    b22 = B[2,2]
    bmin = rtmin * max(abs(b11), abs(b12), abs(b22), rtmin)
    if abs(b11) < bmin
        b11 = b11 < 0 ? -bmin : bmin
    end
    if abs(b22) < bmin
        b22 = b22 < 0 ? -bmin : bmin
    end
    bnorm = max(abs(b11), abs(b12) + abs(b22), safmin)
    bsize = max(abs(b11), abs(b22))
    bscale = one(T) / bsize
    b11 *= bscale
    b12 *= bscale
    b22 *= bscale
    # use method of C van Loan for larger eigval
    binv11 = one(T) / b11
    binv22 = one(T) / b22
    s1 = a11 * binv11
    s2 = a22 * binv22
    if abs(s1) <= abs(s2)
        as12 = a12 - s1 * b12
        as22 = a22 - s1 * b22
        ss = a21 * (binv11 * binv22)
        abi22 = as22 * binv22 - ss * b12
        pp = abi22 / 2
        shift = s1
    else
        as12 = a12 - s2 * b12
        as11 = a11 - s2 * b11
        ss = a21 * (binv11 * binv22)
        abi22 = -ss * b12
        pp = (as11 * binv11 + abi22) / 2
        shift = s2
    end
    qq = ss * as12
    if abs(pp*rtmin) >= one(T)
        discr = (rtmin * pp)^2 + qq * safmin
        r = sqrt(abs(discr)) * rtmax
    else
        if pp^2 + abs(qq) <= safmin
            discr = (rtmax * pp)^2 + qq*safmax
            r = sqrt(abs(discr)) * rtmin
        else
            discr = pp^2 + qq
            r = sqrt(abs(discr))
        end
    end
    if discr >= 0 || r == 0
        sm = pp + (pp < 0 ? -r : r)
        df = pp - (pp < 0 ? -r : r)
        wbig = shift + sm
        wsmall = shift + df
        if abs(wbig) / 2 > max(abs(wsmall), safmin)
            wdet = (a11 * a22 - a12 * a21) * (binv11 * binv22)
            wsmall = wdet / wbig
        end
        if pp > abi22
            wr1 = min(wbig, wsmall)
            wr2 = max(wbig, wsmall)
        else
            wr2 = min(wbig, wsmall)
            wr1 = max(wbig, wsmall)
        end
        wi = zero(T)
    else
        # complex
        wr1 = shift + pp
        wr2 = wr1
        wi = r
    end

    # final scaling
    c1 = bsize * (safmin * max(one(T), ascale))
    c2 = safmin * max(one(T), bnorm)
    c3 = bsize * safmin
    if (ascale <= one(T)) && (bsize <= one(T))
        c4 = min(one(T), (ascale / safmin) * bsize)
    else
        c4 = one(T)
    end
    if (ascale <= one(T)) || (bsize <= one(T))
        c5 = min(one(T), ascale * bsize)
    else
        c5 = one(T)
    end
    wabs = abs(wr1) + abs(wi)
    wsize = max(safmin, c1, fuzzy1 * (wabs * c2 + c3), min(c4, max(wabs, c5)/2))
    if wsize != one(T)
        wscale = one(T) / wsize
        if wsize > one(T)
            scale1 = (max(ascale, bsize) * wscale) * min(ascale, bsize)
        else
            scale1 = (min(ascale, bsize) * wscale) * max(ascale, bsize)
        end
        wr1 *= wscale
        if wi != zero(T)
            wi *= wscale
            wr2 = wr1
            scale2 = scale1
        end
    else
        scale1 = ascale * bsize
        scale2 = scale1
    end
    if wi == 0
        wsize = max(safmin, c1, fuzzy1 * (abs(wr2) * c2 + c3), min(c4, max(abs(wr2), c5)/2))
        if wsize != one(T)
            wscale = one(T) / wsize
            if wsize > one(T)
                scale2 = (max(ascale, bsize) * wscale) * min(ascale, bsize)
            else
                scale2 = (min(ascale, bsize) * wscale) * max(ascale, bsize)
            end
            wr2 *= wscale
        else
            scale2 = ascale * bsize
        end
    end
    return scale1, scale2, wr1, wr2, wi
end

# SVD of real upper triangular 2x2 matrix
# translation of LAPACK dlasv2
function _gsvd_2x2(A::AbstractMatrix{T}) where {T <: AbstractFloat}
    ft = A[1,1]
    gt = A[1,2]
    ht = A[2,2]
    fa = abs(ft)
    ha = abs(ht)
    pmax = 1
    swap = ha > fa
    if swap
        pmax = 3
        ft, ht = ht, ft
        fa, ha = ha, fa
    end
    ga = abs(gt)

    if ga == 0
        ssmin = ha
        ssmax = fa
        clt = one(T)
        crt = one(T)
        slt = zero(T)
        srt = zero(T)
    else
        gasmall = true
        if ga > fa
            pmax = 2
            if (fa/ga) < eps(one(T))
                # very large ga
                gasmall = false
                ssmax = ga
                ssmin = (ha > one(T)) ? (fa / (ga/ha)) : ((fa/ha) * ha)
                clt = one(T)
                slt = ht / gt
                srt = one(T)
                crt = ft / gt
            end
        end
        if gasmall
            # normal case
            d = fa - ha
            # handle infinite ft or ht
            l = (d == fa) ? one(T) : (d / fa)
            m = gt / ft
            t = 2 - l
            mm = m * m
            tt = t * t
            s = sqrt(tt + mm)
            r = (l == 0) ? abs(m) : sqrt(l*l + mm)
            a = (s + r) / 2
            ssmin = ha / a
            ssmax = fa * a
            if mm == 0
                # very tiny m
                if l == 0
                    t = copysign(T(2),ft) * copysign(one(T), gt)
                else
                    t = gs / copysign(d, ft) + m / t
                end
            else
                t = (m / (s + t) + m / (r + l)) * (one(T) + a)
            end
            l = sqrt(t*t + T(4))
            crt = T(2) / l
            srt = t / l
            clt = (crt + srt * m) / a
            slt = (ht / ft) * srt / a
        end
    end
    if swap
        csl, snl = srt, crt
        csr, snr = slt, clt
    else
        csl, snl = clt, slt
        csr, snr = crt, srt
    end
    if pmax == 1
        tsgn = copysign(one(T), csr) * copysign(one(T), csl) * copysign(one(T), A[1,1])
    elseif pmax == 2
        tsgn = copysign(one(T), snr) * copysign(one(T), csl) * copysign(one(T), A[1,2])
    else # pmax == 3
        tsgn = copysign(one(T), snr) * copysign(one(T), snl) * copysign(one(T), A[2,2])
    end
    ssmax = copysign(ssmax, tsgn)
    ssmin = copysign(ssmin, tsgn * copysign(one(T), A[1,1]) * copysign(one(T), A[2,2]))
    return ssmin,ssmax,snr,csr,snl,csl
end
