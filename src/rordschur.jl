using LinearAlgebra: reflector!

function LinearAlgebra._ordschur(T::StridedMatrix{Ty},
                    Z::StridedMatrix{Ty},
                   select::Union{Vector{Bool},BitVector}) where Ty <: Real
    _ordschur!(copy(T), copy(Z), select)
end

function LinearAlgebra._ordschur!(T::StridedMatrix{Ty},
                    Z::StridedMatrix{Ty},
                    select::Union{Vector{Bool},BitVector}) where Ty <: Real
    # suppress most checks since this is an internal function expecting
    # components of a Schur object
    n = size(T,1)

    ks = 0
    pair = false
    for k =1:n
        if pair
            pair = false
        else
            swap = select[k]
            if k < n
                if T[k+1,k] != 0
                    pair = true
                    swap = swap || select[k+1]
                end
            end
            if swap
                ks += 1
                if k != ks
                    ktmp,ks,ok = _trexchange!(T,Z,k,ks)
                    ok || error("exchange $k,$ks failed") # FIXME: throw(something)
                    @mydebug println("after swap ks=$ks")
                end
                if pair
                    ks += 1
                end
            end
        end
    end
    vals = complex.(diag(T))
    for k=1:n-1
        if T[k+1,k] != 0
            y = sqrt(abs(T[k,k+1])) * sqrt(abs(T[k+1,k]))
            vals[k] += y*im
            vals[k+1] -= y*im
        end
    end
    T,Z,vals
end

"""
reorder the real Schur form `T` by an orthogonal similarity transformation so that
the block containing `T[iold,iold]` is moved to `T[inew,inew]`.
Also right-multiply `Z` by the same transformation.
"""
function _trexchange!(T::AbstractMatrix{Ty},Z,iold,inew) where {Ty<:Real}

    n = size(T,1)
    ok = true
    if (n <= 1) || (iold == inew)
        return iold, inew, ok
    end
    # find first rows of specified blocks and their sizes
    if (iold > 1) && (T[iold,iold-1] != 0)
        iold -= 1
    end

    # function to get block size
    nbsize(j) = ((j < n) && (T[j+1,j] != 0)) ? 2 : 1

    nbold = nbsize(iold)
    if (inew > 1) && (T[inew,inew-1] != 0)
        inew -= 1
    end
    nbnew = nbsize(inew)
    if iold == inew
        return iold, inew, ok
    end

    if iold < inew
        @mydebug println("forward swap $iold to $inew sizes $nbold, $nbnew")
        if (nbold == 2) && (nbnew == 1)
            inew -= 1
        elseif (nbold == 1) && (nbnew == 2)
            inew += 1
        end
        ihere = iold
        while ihere < inew
            if nbold ∈ (1,2)
                nbnext = nbsize(ihere+nbold)
                ok = _swap1or2(T,Z,ihere,nbold,nbnext)
                ok || return iold, ihere, ok
                ihere += nbnext
                if (nbold == 2) && (T[ihere+1, ihere] == 0)
                    nbold = 3
                end
            else
                # two 1x1 blocks each of which must be swapped individually
                nbnext = nbsize(ihere+1)
                ok = _swap1or2(T,Z,ihere+1,1,nbnext)
                ok || return iold, ihere, ok
                if nbnext == 1
                    ok = _swap1or2(T,Z,ihere, 1, nbnext)
                    ihere += 1
                else
                    # recompute nbnext in case 2x2 split
                    if T[ihere+2, ihere+1] == 0
                        nbnext = 1
                    end
                    if nbnext == 2
                        # no split
                        ok = _swap1or2(T,Z,ihere,1,nbnext)
                        ok || return iold, ihere, ok
                        ihere += 2
                    else
                        # split
                        ok = _swap1or2(T,Z,ihere,1,1)
                        ok = _swap1or2(T,Z,ihere+1,1,1)
                        ihere += 2
                    end
                end
            end
        end # forward ihere loop
    else
        @mydebug println("reverse swap $iold to $inew sizes $nbold, $nbnew")
        ihere = iold
        while ihere > inew
            if nbold ∈ (1,2)
                nbnext = ((ihere >= 3) && (T[ihere-1,ihere-2] != 0)) ? 2 : 1
                ok = _swap1or2(T,Z,ihere-nbnext,nbnext,nbold)
                ok || return iold, ihere, ok
                ihere -= nbnext
                if (nbold == 2) && (T[ihere+1, ihere] == 0)
                    nbold = 3
                end
            else
                # two 1x1 blocks each of which must be swapped individually
                nbnext = ((ihere >= 3) && (T[ihere-1,ihere-2] != 0)) ? 2 : 1
                ok = _swap1or2(T,Z,ihere-nbnext,nbnext,1)
                ok || return iold, ihere, ok
                if nbnext == 1
                    ok = _swap1or2(T,Z,ihere, nbnext, 1)
                    ihere -= 1
                else
                    # recompute nbnext in case 2x2 split
                    if T[ihere, ihere-1] == 0
                        nbnext = 1
                    end
                    if nbnext == 2
                        # no split
                        ok = _swap1or2(T,Z,ihere-1,2,1)
                        ok || return iold, ihere, ok
                        ihere -= 2
                    else
                        # split
                        ok = _swap1or2(T,Z,ihere,1,1)
                        ok = _swap1or2(T,Z,ihere-1,1,1)
                        ihere -= 2
                    end
                end
            end
        end # reverse ihere loop
    end
    inew = ihere

    return iold,inew,ok
end

# translation of dlaexc
function _swap1or2(T::AbstractMatrix{Ty},Z,j1,n1,n2) where {Ty <: Real}
    @mydebug println(" at $j1 block sizes $n1, $n2")
    n = size(T,1)
    j1+n1 > n && return true
    j2,j3,j4 = j1+1,j1+2,j1+3
    if n1 == 1 && n2 == 1
        T11 = T[j1,j1]
        T22 = T[j2,j2]
        G,_ = givens(T[j1,j2], T22-T11, j1, j2)
        if j3 <= n
            lmul!(G, view(T, :, j3:n))
        end
        rmul!(view(T, 1:j1-1, :), adjoint(G))
        T[j1,j1] = T22
        T[j2,j2] = T11
        Z === nothing || rmul!(Z,adjoint(G))
    else
        # at least one 2x2 block
        # copy diagonal block n1+n2
        nd = n1+n2
        jx = j1+nd-1
        D = T[j1:jx,j1:jx]
        dnorm = norm(D, Inf)
        smallnum = floatmin(Ty) / eps(Ty)
        thresh = max(10 * eps(Ty) * dnorm, smallnum)
        # solve T₁₁ X - X T₂₂ = scale T₁₂
        X, xnorm, scale, ok1 = _syl1or2(view(D,1:n1,1:n1),
                                        view(D,n1+1:nd,n1+1:nd),
                                        view(D,1:n1,n1+1:nd))

        if n1==1 && n2 == 2
            # reflector s.t. [scale, X₁₁ X₁₂] H = [0, 0, _]
            # note backwards ordering
            u = [X[1,2], scale, X[1,1]]
            τ = reflector!(u)
            T11 = T[j1,j1]
            _reflectorYappl!(u,τ,D) # lmul!
            _reflectorYlppa!(u,τ,D) # rmul!
            #lmul!(h',D)
            #rmul!(D,h)
            if max(abs(D[3,1]), abs(D[3,2]), abs(D[3,3] - T11)) > thresh
                return false
            end
            _reflectorYappl!(u,τ,view(T,j1:j1+2,j1:n)) # lmul!
            _reflectorYlppa!(u,τ,view(T,1:j2,j1:j1+2)) # rmul!
            #lmul!(h',view(T,j1:j1+2,j1:n))
            #rmul!(view(T,1:j2,j1:j1+2), h)
            T[j3,j1] = zero(Ty)
            T[j3,j2] = zero(Ty)
            T[j3,j3] = T11
            if !(Z === nothing)
                #rmul!(view(Z,1:n,j1:j1+2), h)
                _reflectorYlppa!(u,τ,view(Z,1:n,j1:j1+2))
            end
        elseif n1 == 2 && n2 == 1
            # reflector s.t. H [ -X₁₁ -X₂₁, scale]ᵀ = [_, 0, 0]ᵀ
            u = [-X[1,1], -X[2,1], scale]
            τ = reflector!(u)
            u[1] = one(Ty)
            T33 = T[j3,j3]
            h = Householder(view(u,2:3),τ)
            lmul!(h',D)
            rmul!(D,h)
            if max(abs(D[1,1] - T33), abs(D[2,1]), abs(D[3,1])) > thresh
                return false
            end
            rmul!(view(T,1:j3,j1:j1+2), h)
            lmul!(h', view(T, j1:j1+2, j2:n))
            T[j1,j1] = T33
            T[j2,j1] = zero(Ty)
            T[j3,j1] = zero(Ty)
            Z === nothing || rmul!(view(Z,1:n,j1:j1+2), h)
        else
            # n1 = 2 && n2 == 2
            # 2 reflectors s.t. H₂H₁ [-X; scale*I₂] is U.T.
            u1 = [-X[1,1], -X[2,1], scale]
            τ1 = reflector!(u1)
            t = -τ1 * (X[1,2] + u1[2] * X[2,2])
            u2 = [-t*u1[2] - X[2,2], -t * u1[3], scale]
            τ2 = reflector!(u2)
            h1 = Householder(view(u1,2:3),τ1)
            h2 = Householder(view(u2,2:3),τ2)
            lmul!(h1',view(D,1:3,:))
            rmul!(view(D,:,1:3),h1)
            lmul!(h2',view(D,2:4,:))
            rmul!(view(D,:,2:4),h2)
            if maximum(abs.([D[3,1],D[3,2],D[4,1],D[4,2]])) > thresh
                return false
            end
            lmul!(h1',view(T,j1:j1+2,j1:n))
            rmul!(view(T,1:j4,j1:j1+2), h1)
            lmul!(h2',view(T,j2:j2+2,j1:n))
            rmul!(view(T,1:j4,j2:j2+2),h2)
            T[j3,j1] = zero(Ty)
            T[j3,j2] = zero(Ty)
            T[j4,j1] = zero(Ty)
            T[j4,j2] = zero(Ty)
            if !(Z === nothing)
                rmul!(view(Z,1:n,j1:j1+2),h1)
                rmul!(view(Z,1:n,j2:j2+2),h2)
            end
        end

        if n2 == 2
            # standardize new 2x2 block T₁₁
            Tx = T[j1:j2,j1:j2]
            G,w1,w2 = _gs2x2!(Tx,2)
            lmul!(G,view(T,j1:j2,j1+2:n))
            rmul!(view(T,1:j1-1,j1:j2),G')
            T[j1:j2,j1:j2] .= Tx
            (Z === nothing) || rmul!(view(Z,1:n,j1:j2),G')
        end
        if n1 == 2
            # standardize new 2x2 block T₂₂
            j3 = j1+n2
            j4 = j3+1
            Tx = T[j3:j4,j3:j4]
            G,w1,w2 = _gs2x2!(Tx,2)
            lmul!(G,view(T,j3:j4,j3+2:n))
            rmul!(view(T,1:j3-1,j3:j4),G')
            T[j3:j4,j3:j4] .= Tx
            (Z === nothing) || rmul!(view(Z,1:n,j3:j4),G')
        end
    end

    return true
end

# solve Tl X - X Tr = scale * B where orders are 1 or 2
# translation of dlasy2
function _syl1or2(Tl::AbstractMatrix{T}, Tr, B) where T
    n1 = size(Tl,1) # checksquare(Tl)
    n2 = size(Tr,1) # checksquare(Tr)
    # B must be n1xn2
    # X is n1xn2
    epst = eps(T)
    smallnum = floatmin(T) / epst
    if n1==1 && n2==1
        τ1 = Tl[1,1] - Tr[1,1]
        β = abs(τ1)
        if β <= smallnum
            τ1 = smallnum
            β = smallnum
            unperturbed = false
        else
            unperturbed = true
        end
        scale = one(T)
        γ = abs(B[1,1])
        if smallnum * γ > β
            scale = one(T) / γ
        end
        X = fill((B[1,1] * scale) / τ1,1,1)
        xnorm = abs(X[1,1])
    elseif n1 == 2 && n2 == 2
        # solve equivalent 4x4 w/ protected complete pivoting
        unperturbed = true
        smin = max(maximum(abs.(Tr)), maximum(abs.(Tl)))
        smin = max(epst*smin, smallnum)
        TT = zeros(T,4,4)
        TT[1,1] = Tl[1,1] - Tr[1,1]
        TT[2,2] = Tl[2,2] - Tr[1,1]
        TT[3,3] = Tl[1,1] - Tr[2,2]
        TT[4,4] = Tl[2,2] - Tr[2,2]

        TT[1,2] = Tl[1,2]
        TT[2,1] = Tl[2,1]
        TT[3,4] = Tl[1,2]
        TT[4,3] = Tl[2,1]

        TT[1,3] = -Tr[2,1]
        TT[2,4] = -Tr[2,1]
        TT[3,1] = -Tr[1,2]
        TT[4,2] = -Tr[1,2]
        bv = [B[1,1], B[2,1], B[1,2], B[2,2]]
        # perform elimination
        ipsv, jpsv = 0,0
        jpiv = [0,0,0,0]
        for i = 1:3
            xmax = zero(T)
            for ip = i:4
                for jp = i:4
                    if abs(TT[ip,jp]) > xmax
                        xmax = abs(TT[ip,jp])
                        ipsv, jpsv = ip,jp
                    end
                end
            end
            if ipsv != i
                tmp = TT[ipsv,:]
                TT[ipsv,:] .= TT[i,:]
                TT[i,:] .= tmp
                bv[i], bv[ipsv] = bv[ipsv], bv[i]
            end
            if jpsv != i
                tmp = TT[:,jpsv]
                TT[:,jpsv] .= TT[:,i]
                TT[:,i] .= tmp
            end
            jpiv[i] = jpsv
            if abs(TT[i,i]) < smin
                unperturbed = false
                TT[i,i] = smin
            end
            for j=i+1:4
                TT[j,i] /= TT[i,i]
                bv[j] -= TT[j,i] * bv[i]
                for k=i+1:4
                    TT[j,k] -= TT[j,i] * TT[i,k]
                end
            end
        end
        if abs(TT[4,4]) < smin
            unperturbed = false
            TT[4,4] = smin
        end
        scale = one(T)
        a = 8smallnum
        if ((a * abs(bv[1]) > TT[1,1]) ||
            (a * abs(bv[2]) > TT[2,2]) ||
            (a * abs(bv[3]) > TT[3,3]) ||
            (a * abs(bv[4]) > TT[4,4]))
            scale = (1 / T(8)) / maximum(abs.(bv))
            bv .*= scale
        end
        xt = zeros(T,4)
        for i=1:4
            k = 5-i
            t = 1 / TT[k,k]
            xt[k] = bv[k] * t
            for j=k+1:4
                xt[k] -= (t * TT[k,j])*xt[j]
            end
        end
        for i=1:3
            jj = jpiv[4-i]
            if jj != 4-i
                xt[4-i],xt[jj] = xt[jj], xt[4-i]
            end
        end
        X = [xt[1] xt[3]; xt[2] xt[4]];
        xnorm = maximum(abs.(xt))
    else
        if n1 == 1 && n2 == 2
            smin = max(epst * max(abs(Tl[1,1]), maximum(abs.(Tr))), smallnum)
            tmp = (Tl[1,1] - Tr[1,1], -Tr[1,2], - Tr[2,1], Tl[1,1] - Tr[2,2])
            btmp = (B[1,1], B[1,2])
        else # n2 == 2 && n2 == 1
            smin = max(epst * max(abs(Tr[1,1]), maximum(abs.(Tl))), smallnum)
            tmp = (Tl[1,1] - Tr[1,1], Tl[2,1], Tl[1,2], Tl[2,2] - Tr[1,1])
            btmp = (B[1,1], B[2,1])
        end
        # solve 2x2 w/ protected complete pivoting
        a, ipiv = findmax(abs.(tmp))
        u11 = tmp[ipiv]
        unperturbed = true
        if a <= smin
            unperturbed = false
            u11 = smin
        end
        locu12, locl21, locu22 = _syl2x2_inds(ipiv)
        u12 = tmp[locu12]
        l21 = tmp[locl21] / u11
        u22 = tmp[locu22] - u12 * l21
        xswap, bswap = _syl2x2_swaps(ipiv)
        if abs(u22) <= smin
            unperturbed = false
            u22 = smin
        end
        if bswap
            btmp = (btmp[2], btmp[1] - l21 * btmp[2])
        else
            btmp = (btmp[1], btmp[2] - l21 * btmp[1])
        end
        scale = one(T)
        if ((2*smallnum) * abs(btmp[2]) > abs(u22)) ||
            ((2*smallnum) * abs(btmp[1]) > abs(u11))
            scale = (1 / T(2)) / max(abs(btmp[1]), abs(btmp[2]))
            btmp = (scale*btmp[1], scale*btmp[2])
        end
        x22 = btmp[2] / u22
        X2 = (btmp[1] / u11 - (u12 / u11) * x22, x22)
        if xswap
            X2 = (X2[2], X2[1])
        end
        if n1 == 1
            X = [X2[1] X2[2]]
            xnorm = abs(X[1,1]) + abs(X[1,2])
        else
            X = [X2[1], X2[2]]
            xnorm = max(abs(X[1,1]), abs(X[2,1]))
        end
    end
    return X, xnorm, scale, unperturbed
end
@inline function _syl2x2_inds(i)
    # locu12, locl21, locu22
    i == 1 && return (3,2,4)
    i == 2 && return (4,1,3)
    i == 3 && return (1,4,2)
    return (2,3,1)
end
@inline function _syl2x2_swaps(i)
    # xswap, bswap
    i == 1 && return (false,false)
    i == 2 && return (false,true)
    i == 3 && return (true,false)
    return (true, true)
end

# TODO: we only need these for order=3, so just write them out inline

# apply shifted reflector from right
@inline function _reflectorYlppa!(x::AbstractVector, τ::Number, A::StridedMatrix)
    require_one_based_indexing(x)
    m, n = size(A)
    if length(x) != n
        throw(DimensionMismatch("reflector has length $(length(x)), which must match the second dimension of matrix A, $n"))
    end
    @inbounds begin
        for j = 1:m
            # dot
            Ajv = A[j, n]
            for i = 1:n-1
                Ajv += A[j, i] * x[i+1]
            end

            Ajv = τ * Ajv

            # ger
            A[j, n] -= Ajv
            for i = 1:n-1
                A[j, i] -= Ajv * x[i+1]'
            end
        end
    end
    return A
end
# apply shifted reflector from left
@inline function _reflectorYappl!(x::AbstractVector, τ::Number, A::StridedMatrix)
    require_one_based_indexing(x)
    m, n = size(A)
    if length(x) != m
        throw(DimensionMismatch("reflector has length $(length(x)), which must match the first dimension of matrix A, $m"))
    end
    @inbounds begin
        for j = 1:n
            # dot
            vAj = A[m, j]
            for i = 1:m-1
                vAj += x[i+1]'*A[i, j]
            end

            vAj = conj(τ)*vAj

            # ger
            A[m, j] -= vAj
            for i = 1:m-1
                A[i, j] -= x[i+1]*vAj
            end
        end
    end
    return A
end
