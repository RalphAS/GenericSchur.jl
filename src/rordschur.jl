using LinearAlgebra: reflector!

function gordschur!(T::StridedMatrix{Ty},
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
                    ok || throw(IllConditionException(k))
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

# Reordering generalized Schur decompositions, Real element type

# needed for delegated RQ
using MatrixFactorizations

function gordschur!(S::StridedMatrix{Ty}, T::StridedMatrix{Ty},
                    Q::StridedMatrix{Ty}, Z::StridedMatrix{Ty},
                    select::Union{Vector{Bool},BitVector}
                    ) where Ty <: AbstractFloat
    n = size(S,1)
    safmin = floatmin(Ty)
    # logic from dtgsen
    ks = 0
    inpair = false
    α = zeros(complex(Ty),n)
    β = zeros(Ty,n)
    for k=1:n
        if inpair
            inpair = false
        else
            doswap = select[k]
            if k < n
                if S[k+1,k] != 0
                    inpair = true
                    doswap = doswap | select[k+1]
                end
            end
            if doswap
                ks += 1
                if k != ks
                    ktmp, ks, ok = _trexchange!(S,T,Q,Z,k,ks)
                    ok || throw(IllConditionException(k))
                    @mydebug println("after swap ks=$ks")
                end
                if inpair
                    ks += 1
                end
            end
        end
    end
    # in case of poor conditioning, eigvals may be significantly modified
    # so recomputation is advised and standardization may be needed
    inpair = false
    for k in 1:n
        if inpair
            inpair = false
        else
            if k < n
                if S[k+1,k] != 0
                    inpair = true
                end
            end
            if inpair
                β1,β2,αr1,αr2,αi = _ggs_2x2(S[k:k+1,k:k+1],T[k:k+1,k:k+1],safmin)
                α[k] = αr1 + im * αi
                α[k+1] = αr2 - im * αi
                β[k] = β1
                β[k+1] = β2
            else
                if copysign(one(Ty), T[k,k]) < 0 # includes -0.0
                    S[k,:] .= -S[k,:]
                    T[k,:] .= -T[k,:]
                    if size(Q,1) > 0
                        Q[:,k] .= -Q[:,k]
                    end
                end
                α[k] = S[k,k]
                β[k] = T[k,k]
            end
        end
    end
    S,T,α,β,Q,Z
end

function _trexchange!(S::AbstractMatrix{Ty},T,Q,Z,iold,inew) where {Ty <: Real}
    # based on dtgexc
    n = size(T,1)
    ok = true
    if n <= 1
        return iold, inew, ok
    end
    # function to get block size
    nbsize(j) = ((j < n) && (S[j+1,j] != 0)) ? 2 : 1
    # find first block size and location
    if (iold > 1) && (S[iold, iold-1] != 0)
            iold -= 1
    end
    nbold = nbsize(iold)
    # find final block size and location
    if (inew > 1) && (S[inew, inew-1] != 0)
        inew -= 1
    end
    nbnew = nbsize(inew)
    if iold == inew
        @mydebug println("no-op call to swap $iold,$inew")
        return iold, inew, true
    end

    if iold < inew
        # WARNING: this branch has not been tested yet
        @mydebug println("forward swap $iold:$(iold+nbold-1)"
            * " with $inew:$(inew+nbnew-1)")
        if nbold == 2 && nbnew == 1
            inew -=1
        end
        if nbold == 1 && nbnew == 2
            inew += 1
        end
        ihere = iold
        while ihere < inew
            # swap with next one below
            if nbold ∈ (1,2)
                nbnext = nbsize(ihere + nbold)
                ok = _swap1or2!(S,T,Q,Z,ihere,nbold,nbnext)
                ok || return iold, ihere, ok
                ihere += nbnext
                # test if 2x2 block breaks into two 1x1
                if nbold == 2 && S[ihere+1,ihere] == 0
                    nbold = 3
                end
            else
                # current block is a pair of 1x1, each must be swapped individually
                nbnext = nbsize(ihere+1)
                ok = _swap1or2!(S,T,Q,Z,ihere+1,1,nbnext)
                ok || return iold, ihere, ok
                if nbnext == 1
                    ok = _swap1or2!(S,T,Q,Z,ihere,1,nbnext)
                    ihere += 1
                else
                    # recompute nbnext in case 2x2 split
                    if S[ihere+2, ihere+1] == 0
                        nbnext = 1
                    end
                    if nbnext == 2
                        # no split
                        ok = _swap1or2!(S,T,Q,Z,ihere,1,nbnext)
                        ok || return iold, ihere, ok
                        ihere += 2
                    else
                        # split
                        ok = _swap1or2!(S,T,Q,Z,ihere,1,1)
                        ok || return iold, ihere, ok
                        ihere += 1
                        ok = _swap1or2!(S,T,Q,Z,ihere,1,1)
                        ok || return iold, ihere, ok
                        ihere += 1
                    end
                end
            end
        end # forward ihere loop
    else # inew < iold
        @mydebug println("reverse swap $iold:$(iold+nbold-1)"
            * " with $inew:$(inew+nbnew-1)")
        ihere = iold
        while ihere > inew
            if nbold ∈ (1,2)
                nbnext = ((ihere >= 3) && (S[ihere-1,ihere-2] != 0)) ? 2 : 1
                ok = _swap1or2!(S,T,Q,Z,ihere-nbnext,nbnext,nbold)
                ok || return iold, ihere, ok
                ihere -= nbnext
                if nbold == 2 && S[ihere+1,ihere] == 0
                    # 2x2 block breaks into pair of 1x1
                    nbold == 3
                    @mydebug println("2x2 block separated")
                end
            else
                # two 1x1 blocks
                nbnext = ((ihere >= 3) && (S[ihere-1,ihere-2] != 0)) ? 2 : 1
                ok = _swap1or2!(S,T,Q,Z,ihere-nbnext,nbnext,1)
                ok || return iold, ihere, ok
                if nbnext == 1
                    # swap 2 1x1s
                    ok = _swap1or2!(S,T,Q,Z,ihere,nbnext,1)
                    ok || return iold, ihere, ok
                    ihere -= 1
                else
                    # recompute in case of 2x2 split
                    if S[ihere, ihere-1] == 0
                        nbnext = 1
                    end
                    if nbnext == 2
                        # no split
                        ok = _swap1or2!(S,T,Q,Z,ihere-1,2,1)
                        ok || return iold, ihere, ok
                        ihere -= 2
                    else
                        # split
                        ok = _swap1or2!(S,T,Q,Z,ihere,1,1)
                        ok || return iold, ihere, ok
                        ihere -= 1
                        ok = _swap1or2!(S,T,Q,Z,ihere,1,1)
                        ok || return iold, ihere, ok
                        ihere -= 1
                    end
                end
            end
        end # reverse ihere loop
    end
    inew = ihere
    return iold, inew, ok
end

"""
`_swap1or2!(A,B,Q,Z,j1,n1,n2)`

Swap adjacent 1x1 or 2x2 blocks in real-Schur pair (A,B) via unitary equivalence transform.
Apply same transform to associated Q,Z.
Reference: Kågström & Poromaa LAWN 87 [K&P]
"""
function _swap1or2!(A::AbstractMatrix{Ty}, B, Q, Z, j1, n1, n2;
                   require_strong=true
) where {Ty <: Real}
    # translation of dtgex2
    @mydebug println(" at $j1 block sizes $n1, $n2")

    n = size(B,1)
    j1+n1 > n && return true
    if n1 == 1 && n2 == 1
        # logic for this case should be the same as the Complex version
        ok = _rtrexch2!(A,B,Q,Z,j1)
        return ok
    else
        # at least one 2x2 block
        m = n1+n2
        je = j1 + m - 1 # last index in working region
        # copy diagonal n1+n2 superblock
        S = A[j1:je,j1:je]
        T = B[j1:je,j1:je]

        dnorm = norm(S, Inf)
        smallnum = floatmin(Ty) / eps(Ty)
        thresh = max(20 * eps(Ty) * dnorm, smallnum)
        threshb = max(20 * eps(Ty) * norm(T, Inf), smallnum)

        # solve Sylvester system (signs follow LAPACK, not K&P)
        # S₁₁ R - L S₂₂ = γ S₁₂
        # T₁₁ R - L T₂₂ = γ T₁₂
        i2 = n1+1
        XR, XL, _, scale, unpert = _syl1or2(view(S,1:n1,1:n1),
                                            view(S,i2:m,i2:m),
                                            view(S,1:n1,i2:m),
                                            view(T,1:n1,1:n1),
                                            view(T,i2:m,i2:m),
                                            view(T,1:n1,i2:m)
        )
        @mydebug unpert || println("numerically singular Sylvester was perturbed")

        # use solution to get swapping transformations
        LI = vcat(-XL, scale * I(n2))
        q_li,_ = qr!(LI)
        q_li = q_li * I(m) # materialization via `Matrix()` may give partial rank

        IR = hcat(scale * I(n1), XR)
        r_ir,z_ir = rq!(IR)
        z_ir = z_ir * I(m)
        # Note that z_ir is transposed wrt K&P

        # perform swap tentatively
        s1 = (q_li' * S) * z_ir'
        t1 = (q_li' * T) * z_ir'

        # Now we need to triangularize B again. Select the more accurate scheme,
        # based on weak stability criterion (posterior |S₂₁|)

        # triangularize B blocks by RQ, apply Q from left to A
        t2a,z_t1a = rq(t1)
        s2a = s1 * z_t1a'
        z2a = z_t1a * z_ir
        brq_a21 = _safe_fnorm(view(s2a,n2+1:m,1:n2))

        # triangularize B blocks by QR, apply Q from right to A
        q_t1b,t2b = qr(t1)
        s2b = q_t1b' * s1
        q2b = q_li * q_t1b
        bqr_a21 = _safe_fnorm(view(s2b,n2+1:m,1:n2))

        # weak stability test
        if min(bqr_a21, brq_a21) > thresh
            @mydebug println("failed weak stability test")
            return false
        end

        # decide which is preferable
        @mydebug println(" |A21| from qr(B): $bqr_a21: from rq(B): $brq_a21")
        if (bqr_a21 <= brq_a21)
            s_new = s2b
            t_new = t2b
            z_new = z_ir
            q_new = q2b
        else
            s_new = s2a
            t_new = t2a
            z_new = z2a
            q_new = q_li
        end

        if require_strong
            resa = _safe_fnorm(A[j1:je,j1:je] - q_new * s_new * z_new)
            resb = _safe_fnorm(B[j1:je,j1:je] - q_new * t_new * z_new)
            if !(resa <= thresh && resb <= threshb)
                @mydebug println("failed strong stability test resA=$resa resB=$resb")
                return false
            end
        end

        s_new[n2+1:m,1:n2] .= zero(Ty)
        # copy back the transformed blocks
        A[j1:je,j1:je] .= s_new
        B[j1:je,j1:je] .= t_new

        # find transformations to standardize the Schur form
        z_tmp = zeros(Ty, m, m)
        q_tmp = zeros(Ty, m, m)
        if n2 > 1
            j2 = j1 + 1
            _, _, csl, snl, csr, snr = _gqz_2x2!(view(A, j1:j2, j1:j2),
                                                view(B, j1:j2, j1:j2))
            q_tmp[1,1] = csl
            q_tmp[2,1] = snl
            q_tmp[1,2] = -snl
            q_tmp[2,2] = csl
            z_tmp[1,1] = csr
            z_tmp[2,1] = snr
            z_tmp[1,2] = -snr
            z_tmp[2,2] = csr
        else
            z_tmp[1,1] = one(Ty)
            q_tmp[1,1] = one(Ty)
        end
        j3 = j1 + n2 # start of new trailing block
        if n1 > 1
            _, _, csl, snl, csr, snr = _gqz_2x2!(view(A, j3:je, j3:je),
                                                view(B, j3:je, j3:je))
            q_tmp[m-1,m-1] = csl
            q_tmp[m,m-1] = snl
            q_tmp[m-1,m] = -snl
            q_tmp[m,m] = csl
            z_tmp[m-1,m-1] = csr
            z_tmp[m,m-1] = snr
            z_tmp[m-1,m] = -snr
            z_tmp[m,m] = csr
        else
            q_tmp[m,m] = one(Ty)
            z_tmp[m,m] = one(Ty)
        end
        # update other parts of active region in A, B (new A₁₂, B₁₂)

        # A[j1:j3-1,j3:je] .= q_tmp[1:n2,1:n2]' * A[j1:j3-1,j3:je]
        # B[j1:j3-1,j3:je] .= q_tmp[1:n2,1:n2]' * B[j1:j3-1,j3:je]
        # A[j1:j3-1,j3:je] .= A[j1:j3-1,j3:je] * z_tmp[n2+1:m,n2+1:m]
        # B[j1:j3-1,j3:je] .= B[j1:j3-1,j3:je] * z_tmp[n2+1:m,n2+1:m]

        ab_tmp = similar(q_tmp, n2, n1)
        mul!(ab_tmp, view(q_tmp', 1:n2, 1:n2), view(A, j1:j3-1, j3:je))
        mul!(view(A, j1:j3-1, j3:je), ab_tmp, view(z_tmp, n2+1:m, n2+1:m))
        mul!(ab_tmp, view(q_tmp', 1:n2, 1:n2), view(B, j1:j3-1, j3:je))
        mul!(view(B, j1:j3-1, j3:je), ab_tmp, view(z_tmp, n2+1:m, n2+1:m))

        # combine transformations
        q_new = q_new * q_tmp
        # z_new came from direct factorization(s), so take adjoint to make it like Z
        z_new = z_new' * z_tmp

        # accumulate into Q and Z
        wantQ = size(Q,1) > 0
        wantZ = size(Z,1) > 0
        if wantQ || wantZ
            qz_tmp = similar(q_new, n, m)
        end
        if wantQ
            # Q[:,j1:je] .= Q[:,j1:je] * q_new
            mul!(qz_tmp, view(Q, :, j1:je), q_new)
            copy!(view(Q, :, j1:je), qz_tmp)
        end
        if wantZ
            # Z[:,j1:je] .= Z[:,j1:je] * z_new
            mul!(qz_tmp, view(Z, :, j1:je), z_new)
            copy!(view(Z, :, j1:je), qz_tmp)
        end
        # update leading and trailing sections of A, B
        if je < n
            A[j1:je,je+1:n] .= q_new' * A[j1:je,je+1:n]
            B[j1:je,je+1:n] .= q_new' * B[j1:je,je+1:n]
        end
        if j1 > 0
            A[1:j1-1,j1:je] .= A[1:j1-1,j1:je] * z_new
            B[1:j1-1,j1:je] .= B[1:j1-1,j1:je] * z_new
        end
    end
    return true
end

"""
`_gqz_2x2!(A,B)`

inplace 2x2 real generalized Schur (QZ), assuming B is UT
returns eigenvalue terms and rotation coefficients
"""
function _gqz_2x2!(A::AbstractMatrix{Ty},B) where {Ty <: Real}
    safmin = floatmin(Ty)
    ulp = eps(Ty)

    # scale A
    anorm = max(abs(A[1,1]) + abs(A[2,1]), abs(A[1,2]) + abs(A[2,2]), safmin)
    ascale = 1 / anorm
    A .*= ascale
    # scale B
    bnorm = max(abs(B[1,1]), abs(B[1,2]) + abs(B[2,2]), safmin)
    bscale = 1 / bnorm
    B .*= bscale

    wi = zero(Ty)
    # check for early deflation
    if abs(A[2,1]) < ulp
        csl = one(Ty)
        snl = zero(Ty)
        csr = one(Ty)
        snr = zero(Ty)
        A[2,1] = zero(Ty)
        B[2,1] = zero(Ty)
    # check for singular B
    elseif abs(B[1,1]) < ulp
        csl,snl,_ = givensAlgorithm(A[1,1], A[2,1])
        G = Givens(1,2,csl,snl)
        csr, snr = one(Ty), zero(Ty)
        rmul!(A, G)
        rmul!(B, G)
        A[2,1] = 0
        B[1:2,1] = 0
    elseif abs(B[2,2]) < ulp
        csr,snr,_ = givensAlgorithm(A[2,2], A[2,1])
        G = Givens(1,2,csr,snr)
        lmul!(G', A)
        lmul!(G', B)
        A[2,1] = 0
        B[2,1:2] .= 0
    else
        scale1, scale2, wr1, wr2, wi = _ggs_2x2(A,B,floatmin(Ty))
        if wi == 0
            # two reals
            # compute H = s * A - w * B
            h1 = scale1 * A[1,1] - wr1 * B[1,1]
            h2 = scale1 * A[1,2] - wr1 * B[1,2]
            h3 = scale1 * A[2,2] - wr1 * B[2,2]
            rr = hypot(h2,h1)
            qq = hypot(scale1 * A[2,1], h3)
            if rr > qq
                # find right rotation to zero H[1,1]
                csr, snr, _ = givensAlgorithm(h2,h1)
            else
                # zero H[2,1]
                csr, snr, _ = givensAlgorithm(h3, scale1 * A[2,1])
            end
            G = Givens(1,2,csr,snr)
            rmul!(A,G)
            rmul!(B,G)
            snr = -snr
            t1 = max(abs(A[1,1]) + abs(A[1,2]), abs(A[2,1]) + abs(A[2,2]))
            t2 = max(abs(A[1,1]) + abs(A[1,2]), abs(A[2,1]) + abs(A[2,2]))
            if scale1 * t1 >= abs(wr1) * t2
                # left rotation to zero B[2,1]
                csl,snl,_ = givensAlgorithm(B[1,1], B[2,1])
            else
                # left rotation to zero A[2,1]
                csl,snl,_ = givensAlgorithm(A[1,1], A[2,1])
            end
            G = Givens(1,2,csl,snl)
            lmul!(G,A)
            lmul!(G,B)
            A[2,1] = 0
            B[2,1] = 0
        else
            # complex pair
            ssmin, ssmax, snr, csr, snl, csl = _gsvd_2x2(B)
            G = Givens(1,2,csr,snr)
            rmul!(A,G')
            rmul!(B,G')
            G = Givens(1,2,csl,snl)
            lmul!(G,A)
            lmul!(G,B)
            B[2,1] = 0
            B[1,2] = 0
        end
    end

    # undo scaling
    A .*= anorm
    B .*= bnorm
    if wi == 0
        α = [A[1,1]+zero(Ty)*im, A[2,2]+zero(Ty)*im]
        β = [B[1,1], B[2,2]]
    else
        α1 = anorm * (wr1 + wi * im) / scale1 / bnorm
        α = [α1, conj(α1)]
        β = ones(Ty,2)
    end
    return α, β, csl, snl, csr, snr
end

"""
_syl1or2(A,B,C,D,E,F)

solve A R - L B = scale * C; D R - L E = scale * F
where orders are small and coeffts are generalized Schur forms
"""
function _syl1or2(A::AbstractMatrix{Ty}, B, C, D, E, F) where {Ty}
    # eventually translate portions of dtgsy2
    # use naive Kronecker scheme until we have time to do this...
    n1 = checksquare(A)
    n2 = checksquare(B)
    dimsok = ((size(F) == (n1,n2)) && (size(C) == (n1,n2))
             && (n1 == checksquare(D)) && (n2  == checksquare(E)))
    dimsok || throw(DimensionMismatch("args have incompatible dimensions"))

    K = [kron(I(n2), A) kron(-B', I(n1)); kron(I(n2), D) kron(-E', I(n1))]
    rl = vcat(vec(C), vec(F))
    rl,unperturbed,scale = _xlinsolve!(K, rl)
    xnorm = maximum(abs, rl)
    nn = n1 * n2
    R = reshape(rl[1:nn], n1, n2)
    L = reshape(rl[nn+1:end], n1, n2)
    return R, L, xnorm, scale, unperturbed
end

# this should not be needed (method used for complex should suffice)
# but LAPACK organizes the logic differently from complex, so we follow
# them for now
# TODO: figure out why simple call to general version failed
function _rtrexch2!(A::AbstractMatrix{Ty},B,Q,Z,j1;
                   require_strong=true, throws=false
) where {Ty <: AbstractFloat}
    # copy the block to try the swap
    S = A[j1:j1+1,j1:j1+1]
    T = B[j1:j1+1,j1:j1+1]
    # compute threshold for acceptance
    smlnum = safemin(Ty) / eps(Ty)
    W = hcat(S,T)
    sa = _safe_fnorm(vec(W))
    thresh = max(Ty(20)*eps(Ty)*sa, smlnum)

    # compute Givens rotations which would swap blocks
    # and tentatively do it
    f = S[2,2] * T[1,1] - T[2,2] * S[1,1]
    g = S[2,2] * T[1,2] - T[2,2] * S[1,2]
    sa = abs(S[2,2]) * abs(T[1,1])
    sb = abs(S[1,1]) * abs(T[2,2])
    cz, sz, _ = givensAlgorithm(f, g)
    sz, cz = cz, sz
    Gz1 = Givens(1,2,Ty(cz),-sz)
    rmul!(S,adjoint(Gz1))
    rmul!(T,adjoint(Gz1))
    if sa >= sb
        cq, sq, _ = givensAlgorithm(S[1,1],S[2,1])
    else
        cq, sq, _ = givensAlgorithm(T[1,1],T[2,1])
    end
    Gq1 = Givens(1,2,Ty(cq),sq)
    lmul!(Gq1,S)
    lmul!(Gq1,T)
    # weak stability test: subdiags <= O( ϵ norm((S,T),"F"))
    ws = abs(S[2,1]) + abs(T[2,1])
    if ws > thresh
        @mydebug println("failed weak stability test; subdiag norm = $ws")
        throws && throw(IllConditionException(j1))
        return false
    end

    if require_strong
        # Strong stability test
        # Supposedly equiv. to
        # |[A-Qᴴ*S*Z, B-Qᴴ*T*Z]| <= O(1) ϵ |[A,B]| in F-norm
        Ss = copy(S)
        Ts = copy(T)
        Gz2 = Givens(1,2,Ty(cz),sz)
        rmul!(Ss,adjoint(Gz2))
        rmul!(Ts,adjoint(Gz2))
        Gq2 = Givens(1,2,Ty(cq),-sq)
        lmul!(Gq2,Ss)
        lmul!(Gq2,Ts)
        Ss .-= A[j1:j1+1,j1:j1+1]
        Ts .-= B[j1:j1+1,j1:j1+1]
        W = hcat(Ss,Ts)
        ss = _safe_fnorm(vec(W))
        if ss > thresh
            @mydebug println("failed strong stability test resid=$ss")
            throws && throw(IllConditionException(j1))
            return false
        end
    end
    Gz = Givens(j1,j1+1,Ty(cz),-sz)
    rmul!(A,adjoint(Gz))
    rmul!(B,adjoint(Gz))
    Gq = Givens(j1,j1+1,Ty(cq),sq)
    lmul!(Gq,A)
    lmul!(Gq,B)
    A[j1+1,j1] = zero(Ty)
    B[j1+1,j1] = zero(Ty)
    rmul!(Z,adjoint(Gz))
    rmul!(Q,adjoint(Gq))
    return true
end
