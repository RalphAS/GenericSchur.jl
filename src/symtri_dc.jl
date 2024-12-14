# divide-and-conquer algorithm for real symmetric tridiagonals

using LinearAlgebra: DivideAndConquer

# based on LAPACK xSTEDC
function _gschur!(A::SymTridiagonal{T},
                  alg::DivideAndConquer,
                  Z::Union{Nothing, AbstractArray} = nothing;
                  maxiter=30*size(A,1), nqrmax=25) where {T}
    n = size(A,1)
    if n == 0
        return A
    end
    if n == 1
        if Z !== nothing
            Z[1,1] = one(T)
        end
        return
    end
    # following STEDC, we could fall back to Pal-Walker-Kahen scheme for eigvals only

    small_size = nqrmax

    d = A.dv
    e = A.ev
    n = length(d)
    wantZ = (Z !== nothing) && (size(Z,1) > 0)
    if wantZ
        nz = size(Z,2)
        (nz == n) || throw(DimensionMismatch("second dimension of Z must match tridiagonal A"))
    end

    epsT = eps(T)
    if n <= small_size
        @mydebug println("QR for full problem")
        _gschur!(A, QRIteration(), Z)
    else
        # if any subdiagonals are negligible, we will
        # subdivide the matrix into independent problems
        istart = 1
        while (istart <= n)
            ifinish = istart
            while ifinish < n
                tiny = epsT * sqrt(abs(d[ifinish])) * sqrt(abs(d[ifinish + 1]))
                if abs(e[ifinish]) <= tiny
                    break
                end
                ifinish += 1
            end
            m = ifinish - istart + 1
            if m == 1
                istart = ifinish + 1
                continue
            end
            dvw = view(d, istart:ifinish)
            evw = view(e, istart:ifinish-1)
            if m > small_size
                # scale
                t1,_ = findmax(abs, dvw)
                t2,_ = findmax(abs, evw)
                orgnrm = max(t1,t2)
                dvw .*= (1/orgnrm)
                evw .*= (1/orgnrm)
                Zwrk = similar(Z,m,m)
                Zwrk .= I(m)
                _dcschur_core!(dvw, evw, Zwrk, m, nqrmax)
                if wantZ
                    # multiply back into Z
                    Ztmp = copy(view(Z, 1:n, istart:ifinish))
                    mul!(view(Z, 1:n, istart:ifinish), Ztmp, Zwrk)
                end
                # unscale
                dvw .*= orgnrm
            else
                # solve subproblem
                # using workspace for Z
                # CHECKME: do we have the same constraint on column(?) dim as STEQR?
                #          otherwise maybe pass Z or a view directly
                if wantZ
                    Zwrk = similar(Z,m,m)
                    Zwrk .= I(m)
                else
                    Zwrk = nothing
                end
                @mydebug println("QR for isolated problem $istart:$ifinish")
                _gschur!(SymTridiagonal(dvw, evw), QRIteration(), Zwrk)
                if Z !== nothing
                    # multiply back into Z
                    Ztmp = copy(view(Z, 1:n, istart:ifinish))
                    mul!(view(Z, 1:n, istart:ifinish), Ztmp, Zwrk)
                end
            end
            istart = ifinish + 1
        end
    end
end

# based on LAPACK.dlaed0
function _dcschur_core!(d, e, Z, n, nqrmax)
    # to avoid index confusion we only allow for identity Z here
    # divide matrix into subproblems
    # build a size tree
    nsubprobs = 1 # SUBPBS
    itree = [n] # IWORK[1...]
    tlevels = 0 # TLVLS
    while itree[nsubprobs] > nqrmax
        resize!(itree, 2*nsubprobs)
        for j in nsubprobs:-1:1
            itree[2*j] = itree[j] >> 1
            itree[2*j-1] = (itree[j] + 1) >> 1
        end
        tlevels += 1
        nsubprobs *= 2
    end
    @mydebug println("nsubprobs = $nsubprobs itree = $itree")
    for j in 2:nsubprobs
        itree[j] = itree[j] + itree[j-1]
    end

    # do the rank-1 cuts to separate the leaves
    for i in 1:nsubprobs-1
        i1 = itree[i] + 1
        i2 = i1 - 1
        d[i2] -= abs(e[i2])
        d[i1] -= abs(e[i2])
    end

    indexq = zeros(Int,n)
    for i in 0:nsubprobs-1
        if i == 0
            submat = 1
            matsize = itree[1]
        else
            submat = itree[i] + 1
            matsize = itree[i+1] - itree[i]
        end
        Qwrk = similar(Z, matsize, matsize)
        Qwrk .= I(matsize)
        iend = submat + matsize - 1
        @mydebug println("QR for subproblem $submat:$iend")
        dvw = view(d, submat:iend)
        evw = view(e, submat:iend-1)
        _gschur!(SymTridiagonal(dvw, evw), QRIteration(), Qwrk)
        iend = submat + matsize - 1
        copy!(view(Z,submat:iend,submat:iend), Qwrk)
        # mul!(view(Qstore,1:qsiz,submat:iend), view(Z,1:qsiz,submat:iend), Qwrk)
        k = 1
        for j in submat:itree[i+1]
            indexq[j] = k
            k += 1
        end
    end

    # successively merge up the tree
    curlevel = 1
    while nsubprobs > 1
        for i in 0:2:nsubprobs-2
            if i == 0
                submat = 1
                matsize = itree[2]
                msd2 = itree[1]
                # curprob = 0
            else
                submat = itree[i] + 1
                matsize = itree[i+2] - itree[i]
                # claim integer division rounds up in Fortran
                msd2 = (matsize + 1) >> 1
                # curprob += 1
            end
            iend = submat + matsize - 1
            dvw = view(d, submat:iend)
            # evw = view(e, submat:iend-1)
            Qvw = view(Z, submat:iend, submat:iend) # CHECKME
            ρ = e[submat + msd2 - 1]
            @mydebug println("merging submat $submat:$iend at ncut=$msd2")
            _dcmerge!(dvw, Qvw, view(indexq, submat:iend), ρ, msd2)
            itree[i >> 1 + 1] = itree[i+2]
        end
        nsubprobs = nsubprobs >> 1
        curlevel += 1
    end
end

# based on DLAED1
"""
merge lower order eigensystems of size `ncut` and `(n - ncut)`
"""
function _dcmerge!(d, Q, indexq, ρ, ncut)
    n = size(d,1)
    # last row of Q₁ and first row of Q₂
    zpp1 = ncut + 1
    z = vcat(view(Q, ncut:ncut, 1:ncut)', view(Q, zpp1:zpp1, zpp1:n)')
    k, w, ρmod, λ, indexc, colcounts, Q2 = _dcdeflate!(d, Q, indexq, ρ, z, ncut)
    if k != 0
        @mydebug println("after deflation k = $k colcounts = $(colcounts[1:4]) pairs $(colcounts[5])")
        # solve secular equation
        _dcsecular!(k, ncut, d, Q, ρmod, λ, Q2, indexc, colcounts,w)
        # prepare indexq sorting permutation
        # or just use sortperm!(indexq, d)?
        #_mergeperm!(indexq, k, n-k, d, true, false)
        sortperm!(indexq, d)
    else
        @mydebug println("trivial deflation")
        indexq .= 1:n
    end
    return
end

# mutates d, Q, indexq,
# based on DLAED2
function _dcdeflate!(d::AbstractVector{T}, Q, indexq, ρ, z, n1) where {T}
    n = size(d,1)
    n2 = n - n1
    n1p1 = n1 + 1

    # renormalize z
    t = 1 / sqrt(T(2))
    rmul!(z, t)
    if ρ < 0
        rmul!(view(z, n1p1:n), -1)
    end
    ρ = abs(2 * ρ)
    indexq[n1p1:n] .+= n1
    w = zeros(T,n)

    # re-integrate the deflated parts from last pass
    λ = d[indexq]
    indexc = sortperm(λ)
    index = indexq[indexc]
    # indexp will be used to keep track of deflated pairs: first k entries are not deflated
    indexp = zeros(Int, n)
    # calculate deflation tolerance
    zmax, _ = findmax(abs, z)
    dmax, _ = findmax(abs, d)
    tol = 8 * eps(T) * max(zmax, dmax)
    if ρ * zmax <= tol
        # if rank-1 modifier is tiny, just need to reorganize Q so columns
        # correspond to d
        Qtmp = similar(Q)
        k = 0
        for j in 1:n
            i = index[j]
            copy!(view(Qtmp,1:n,j), view(Q, 1:n, i))
            λ[j] = d[i]
        end
        copy!(Q, Qtmp)
        copy!(d, λ)
        return k, w, ρ, λ, indexc, Int[], [similar(Q,0,0)]
    end

    # coltype[j] indicates that column j is
    # 1) nonzero in upper part,
    # 2) dense
    # 3) nonzero in lower part
    # 4) deflated
    coltype = ones(Int, n)
    coltype[n1p1:n] .= 3

    # find the number of equal eigenvalues
    # compute reflector s.t. corresponding components of Z are 0 in new basis
    k = 0
    k2 = n + 1
    bdone = false
    nclose = 0
    jj = 0
    local pj
    for j in 1:n
        jj = j
        nj = index[j]
        if ρ * abs(z[nj]) <= tol
            # deflate because of small element in z
            k2 -= 1
            coltype[nj] = 4
            indexp[k2] = nj
            @mydebug println("deflate small z[$nj] to $k2")
            if j == n
                bdone = true
                break
            end
        else
            pj = nj
            break
        end
    end
    if !bdone
        while jj < n
            jj += 1
            nj = index[jj]
            if ρ * abs(z[nj]) <= tol
                # deflate because of small element in z
                k2 -= 1
                coltype[nj] = 4
                indexp[k2] = nj
                @mydebug println("deflate small z[$nj] to $k2")
            else
                # check if eigvals are close enough to deflate
                s = z[pj]
                c = z[nj]
                τ = hypot(c, s)
                t = d[nj] - d[pj]
                c /= τ
                s = -s / τ
                if abs(t * c * s) <= tol
                    nclose += 1
                    # deflation is possible
                    z[nj] = τ
                    z[pj] = 0
                    reclass = false
                    if coltype[nj] != coltype[pj]
                        reclass = true
                        coltype[nj] = 2
                    end
                    coltype[pj] = 4
                    g = LinearAlgebra.Givens(pj, nj, c, s)
                    rmul!(Q, g')
                    t = d[pj] * c^2 + d[nj] * s^2
                    d[nj] = d[pj] * s^2 + d[nj] * c^2
                    d[pj] = t
                    k2 -= 1
                    i = 1

                    k2s = 0 # diagnostic only
                    if k2 + i > n
                        indexp[k2 + i - 1] = pj
                        k2s = k2+i-1
                    else
                        while k2 + i <= n
                            if d[pj] < d[indexp[k2+i]]
                                indexp[k2+i-1] = indexp[k2+i]
                                indexp[k2+i] = pj
                                k2s = k2+i
                                i += 1
                            else
                                indexp[k2 + i - 1] = pj
                                k2s = k2+i-1
                                break
                            end
                        end
                    end
                    @mydebug println("deflate close z[$nj] z[$pj] to $k2s")
                    pj = nj
                else
                    k += 1
                    λ[k] = d[pj]
                    w[k] = z[pj]
                    indexp[k] = pj
                    pj = nj
                end # if deflating
            end
        end
    end

    # record last eigenvalue
    k += 1
    λ[k] = d[pj]
    w[k] = z[pj]
    indexp[k] = pj
    colcounts = [count(x -> x == j, coltype) for j in 1:4]
    push!(colcounts, nclose)
    psm = cumsum([1,colcounts[1:3]...])
    k = n - colcounts[4]

    # fill out indexc to sort by type
    for j in 1:n
        js = indexp[j]
        ct = coltype[js]
        index[psm[ct]] = js
        indexc[psm[ct]] = j
        psm[ct] += 1
    end

    # sort eigenvalues and eigenvectors
    i = 1
    Q2 = [similar(Q,n1,colcounts[1]), similar(Q,n1,colcounts[2]), similar(Q,n2,colcounts[2]),
        similar(Q,n2,colcounts[3]), similar(Q,n,colcounts[4])]
    #  iq2 = n1*(c1+c2)
    for j in 1:colcounts[1]
        js = index[i]
        copy!(view(Q2[1],:,j), view(Q,1:n1,js))
        z[i] = d[js]
        i += 1
    end
    # iq1 = n1*c1+1
    for j in 1:colcounts[2]
        js = index[i]
        copy!(view(Q2[2],:,j), view(Q,1:n1,js))
        copy!(view(Q2[3],:,j), view(Q,n1+1:n1+n2,js))
        z[i] = d[js]
        i += 1
    end
    # iq1 = n1*(c1+c2)
    for j in 1:colcounts[3]
        js = index[i]
        copy!(view(Q2[4],:,j), view(Q,n1+1:n1+n2,js))
        z[i] = d[js]
        i += 1
    end
    for j in 1:colcounts[4]
        js = index[i]
        copy!(view(Q2[5],:,j), view(Q,1:n,js))
        z[i] = d[js]
        i += 1
    end

    if k < n
        # put sorted, deflated eigenpairs in last n-k slots of Q and d
        for j in 1:colcounts[4]
            copy!(view(Q,:,k+j), view(Q2[5],:,j))
            d[k+j] = z[k+j]
        end
    end
    return k, w, ρ, λ, indexc, colcounts, Q2
end

# based on DLAED3
function _dcsecular!(k, n1, d::AbstractVector{T}, Q, ρ, λ, Q2, indexc, colcounts, w,) where {T}
    n = size(d,1)
    # we assume IEEE765 so skip the missing guard digit patch
    if k == 0
        return nothing
    end
    for j in 1:k
        # stores eigvec info in column of Q
        d[j], converged = _dcroot!(k, j, view(λ,1:k), view(w,1:k), view(Q,1:k,j), ρ)
    end
    s0 = similar(w,k)
    if k == 2
        for j in 1:k
            w[1] = Q[1,j]
            w[2] = Q[2,j]
            Q[1,j] = w[indexc[1]]
            Q[2,j] = w[indexc[2]]
        end
    elseif k > 1
        copy!(view(s0,1:k,1), view(w,1:k))
        for j in 1:k
            w[j] = Q[j,j]
        end
        for j in 1:k
            for i in 1:j-1
                w[i] = w[i] * (Q[i,j] / (λ[i] - λ[j]))
            end
            for i in j+1:k
                w[i] = w[i] * (Q[i,j] / (λ[i] - λ[j]))
            end
        end
        for i in 1:k
            w[i] = sign( s0[i]) * sqrt(-w[i])
        end
        # compute eigenvectors of modified rank-1
        for j in 1:k
            for i in 1:k
                s0[i] = w[i] / Q[i,j]
            end
            t = norm(s0)
            for i in 1:k
                Q[i,j] = s0[indexc[i]] / t
            end
        end
    end
    # compute updated eigenvectors
    n2 = n - n1
    n12 = colcounts[1] + colcounts[2] # components in first part
    n23 = colcounts[2] + colcounts[3] # components in second part
    S = copy(view(Q, colcounts[1]+1:colcounts[1]+n23, 1:k))
    if n23 > 0
        mul!(view(Q, n1+1:n, 1:k), Q2[3], view(S, 1:colcounts[2], :))
        mul!(view(Q, n1+1:n, 1:k), Q2[4], view(S, colcounts[2]+1:n23, :), true, true)
    else
        Q[n1+1:n, 1:k] .= 0
    end
    S = copy(view(Q, 1:n12, 1:k))
    if n12 > 0
        mul!(view(Q, 1:n1, 1:k), Q2[1], view(S, 1:colcounts[1], :))
        mul!(view(Q, 1:n1, 1:k), Q2[2], view(S, colcounts[1]+1:n12,:), true, true)
    else
        Q[1:n1,1:k] .= 0
    end
    return nothing
end

# solve for a single root of the secular equation
# based on DLAED4
# reference: R-C. Li, LAWN89 (1993)
# stores eigvec info in δ, returns eigval and convergence flag
function _dcroot!(n,i,d,z,δ,ρ)
    maxit = 30
    if n == 1
        λ = d[1] + ρ * z[1]^2
        δ[1] = 1
        return λ, true
    elseif n == 2
        λ = _dcroot2!(i,d,z,δ,ρ)
        return λ, true
    end
    T = eltype(d)
    epsT = eps(T)
    ρinv = 1 / ρ
    local dltlb, dltub

    # subfunctions for case i == n
    function initialize_n(ii, midpt)
        ψ = zero(T)
        for j in 1:n-2
            ψ += z[j]^2 / δ[j]
        end
        c = ρinv + ψ
        w = c + z[ii]^2 / δ[ii] + z[n]^2 / δ[n]
        if w <= 0
            t = z[n-1]^2 / (d[n] - d[n-1] + ρ) + z[n]^2 / ρ
            if c <= t
                τ = ρ
            else
                del = d[n] - d[n-1]
                a = -c * del + z[n-1]^2 + z[n]^2
                b = z[n]^2 * del
                if a < 0
                    τ = 2 * b / (sqrt(a^2 + 4 * b * c) - a)
                else
                    τ = (a + sqrt(a^2 + 4 * b * c)) / (2 * c)
                end
            end
            # it can be proved that d[n] + ρ/2 <= λ[n] < d[n] + τ <= d[n] + ρ
            dltlb = midpt
            dltub = ρ
        else
            del = d[n] - d[n-1]
            a = -c * del + z[n-1]^2 + z[n]^2
            b = z[n]^2 * del
            if a < 0
                τ = 2 * b / (sqrt(a^2 + 4 * b * c) - a)
            else
                τ = (a + sqrt(a^2 + 4 * b * c)) / (2 * c)
            end
            # it can be proved that d[n] < d[n] + τ < d[n] + ρ / 2
            dltlb = zero(T)
            dltub = midpt
        end
        return w, τ, dltlb, dltub
    end

    function update_w_n(ii, τ)
        # evaluate ψ and derivative
        dψ = zero(T)
        ψ = zero(T)
        erretm = zero(T)
        for j in 1:ii
            t = z[j] / δ[j]
            ψ += z[j] * t
            dψ += t^2
            erretm += ψ
        end
        erretm = abs(erretm)
        # evaluate ϕ and derivative
        t = z[n] / δ[n]
        ϕ = z[n] * t
        dϕ = t^2

        erretm = 8 * (-ϕ - ψ) + erretm - ϕ + ρinv + abs(τ) * (dψ + dϕ)
        w = ρinv + ϕ + ψ
        parts_n = (; dψ, dϕ)
        return w, erretm, parts_n
    end

    function next_step_n(w, τ, parts)
        dψ, dϕ = parts.dψ, parts.dϕ
        c = abs(w - δ[n-1] * dψ - δ[n] * dϕ)
        a = (δ[n-1] + δ[n]) * w - δ[n-1] * δ[n] * (dψ + dϕ)
        b = δ[n-1] * δ[n] * w
        if c == 0
            η = dltub - τ
        elseif a >= 0
            η = (a + sqrt(abs(a^2 - 4 * b * c))) / (2 * c)
        else
            η = 2 * b / (a - sqrt(abs(a^2 - 4 * b * c)))
        end
        # η should have opposite sign from w;
        # if violated by roundoff error take a Newton step to fix
        if w * η > 0
            η = -w / (dψ + dϕ)
        end
        t = τ + η
        if t > dltub || t < dltlb
            if w < 0
                η = (dltub - τ) / 2
            else
                η = (dltlb - τ) /2
            end
        end
        return η
    end

    # subfunctions for general i
    function initialize(midpt, del)
        ip1 = i + 1
        ψ = zero(T)
        for j in 1:i-1
            ψ = ψ + z[j]^2 / δ[j]
        end
        ϕ = zero(T)
        for j in n:-1:i+2
            ϕ = ϕ + z[j]^2 / δ[j]
        end
        c = ρinv + ψ + ϕ
        w = c + z[i]^2 / δ[i] + z[ip1]^2 / δ[ip1]

        origin_at_i = w > 0
        if origin_at_i
            # i-th eigval is in (d[i], (d[i]+d[i+1])/2) dᵢ₊₁
            a = c * del + z[i]^2 + z[ip1]^2
            b = z[i]^2 * del
            if a > 0
                τ = 2 * b / (a + sqrt(abs(a^2 - 4 * b * c)))
            else
                τ = (a - sqrt(abs(a^2 - 4 * b * c))) / (2 * c)
            end
            dltlb = zero(T)
            dltub = midpt
        else
            # i-th eigval is in ((d[i] + d[i+1]) / 2, d[i+1])
            # origin is at d[i+1]
            a = c * del - z[i]^2 - z[ip1]^2
            b = z[ip1]^2 * del
            if a < 0
                τ = 2 * b / (a - sqrt(abs(a^2 + 4 * b * c)))
            else
                τ = -(a + sqrt(abs(a^2 + 4 * b * c))) / (2 * c)
            end
            dltlb = -midpt
            dltub = zero(T)
        end
        return w, τ, dltlb, dltub, origin_at_i
    end

    function update_w(ii, τ, ηq)
        iip1 = ii+1
        iim1 = ii-1
        # evaluate ψ and derivative
        dψ = zero(T)
        ψ = zero(T)
        erretm = zero(T)
        for j in 1:iim1
            t = z[j] / δ[j]
            ψ += z[j] * t
            dψ += t^2
            erretm += ψ
        end
        erretm = abs(erretm)
        # evaluate ϕ and derivative
        ϕ = zero(T)
        dϕ = zero(T)
        for j in n:-1:iip1
            t = z[j] / δ[j]
            ϕ += z[j] * t
            dϕ += t^2
            erretm += ϕ
        end
        w0 = ρinv + ϕ + ψ
        t = z[ii] / δ[ii]
        dw = dψ + dϕ + t^2
        t = z[ii] * t
        w = ρinv + ϕ + ψ + t
        erretm = 8 * (ϕ - ψ) + erretm + 2 * ρinv + 3 * abs(t) + abs(τ + ηq) * dw
        parts = (;ψ, ϕ, dψ, dϕ, dw)
        return w, erretm, parts, w0
    end

    function next_step(ii, niter, w, τ, parts, switch3, origin_at_i, switch)
        dψ, dϕ = parts.dψ, parts.dϕ
        ψ, ϕ = parts.ψ, parts.ϕ
        dw = parts.dw
        ip1 = i + 1
        iip1 = ii+1
        iim1 = ii-1
        zz = similar(d, 3)
        if !switch3
            if !switch
                if origin_at_i
                    c = w - δ[ip1] * dw - (d[i] - d[ip1]) * (z[i] / δ[i])^2
                else
                    c = w - δ[i] * dw - (d[ip1] - d[i]) * (z[ip1] / δ[ip1])^2
                end
            else
                t = z[ii] / δ[ii]
                if origin_at_i
                    dψ += t^2
                else
                    dϕ += t^2
                end
                c = w - δ[i] * dψ - δ[ip1] * dϕ
            end
            a = (δ[i] + δ[ip1]) * w - δ[i] * δ[ip1] * dw
            b = δ[i] * δ[ip1] * w
            if c == 0
                if a == 0
                    if origin_at_i
                        a = z[i]^2 + δ[ip1]^2 * (dψ + dϕ)
                    else
                        a = z[ip1]^2 + δ[i]^2 * (dψ + dϕ)
                    end
                end
                η = b / a
            elseif a <= 0
                η = (a - sqrt(abs(a^2 - 4*b*c))) / (2 * c)
            else
                η = 2 * b / (a + sqrt(abs(a^2 - 4*b*c)))
            end
        else
            # interpolation using 3 most relevant poles
            t = ρinv + ψ + ϕ
            if origin_at_i
                t1 = (z[iim1] / δ[iim1])^2
                c = t - δ[iip1] * (dψ + dϕ) - (d[iim1] - d[iip1]) * t1
                zz[1] = z[iim1]^2
                zz[3] = δ[iip1]^2 * ((dψ - t1) + dϕ)
            else
                t1 = (z[iip1] / δ[iip1])^2
                c = t - δ[iim1] * (dψ + dϕ) - (d[iip1] - d[iim1]) * t1
                zz[1] = δ[iim1]^2 * (dψ + (dϕ - t1))
                zz[3] = z[iip1]^2
            end
            zz[2] = z[ii]^2
            η, converged = _dcnewton(niter, origin_at_i, c, view(δ,iim1:iip1), zz, w)
        end
        if w * η > 0
            η = -w / dw
        end
        t = τ + η
        if t > dltub || t < dltlb
            if w < 0
                η = (dltub - τ) / 2
            else
                η = (dltlb - τ) / 2
            end
        end
        return η
    end

    if i == n
        ii = n-1
        niter = 1
        midpt = ρ / 2
        δ .= (d .- d[i]) .- midpt

        w, τ, dltlb, dltub = initialize_n(ii, midpt)
        δ .= (d .- d[i]) .- τ

        w, erretm, parts_n = update_w_n(ii, τ)

        # test for convergence
        if abs(w) < epsT * erretm
            λ = d[i] + τ
            return λ, true
        end

        if w <= 0
            dltlb = max(dltlb, τ)
        else
            dltub = min(dltub, τ)
        end

        niter += 1

        η = next_step_n(w, τ, parts_n)
        δ .= δ .- η
        τ += η
        w, erretm, parts_n = update_w_n(ii, τ)
        iter = niter + 1
        for niter in iter:maxit
            # test for convergence
            if abs(w) < epsT * erretm
                λ = d[i] + τ
                return λ, true
            end

            if w <= 0
                dltlb = max(dltlb, τ)
            else
                dltub = min(dltub, τ)
            end
            η = next_step_n(w, τ, parts_n)
            δ .= δ .- η
            τ += η
            w, erretm, parts_n = update_w_n(ii, τ)
        end
        # if we get here, convergence failed
        λ = d[i] + τ
        return λ, false
    else
        # i ∈ [3..n-1]
        niter = 1
        ip1 = i+1
        # initial guess
        del = d[ip1] - d[i]
        midpt = del / 2
        δ .= (d .- d[i]) .- midpt
        w, τ, dltlb, dltub, origin_at_i = initialize(midpt, del)
        ii = origin_at_i ? i : i+1
        dorg = d[ii]
        δ .= (d .- dorg) .- τ
        w, erretm, parts, w0 = update_w(ii, τ, zero(T))
        switch3 = false
        if origin_at_i
            if w0 < 0
                switch3 = true
            end
        else
            if w0 > 0
                switch3 = true
            end
        end
        if ii == 1 || ii == n
            # not enough points for 3
            switch3 = false
        end

        # test for convergence
        if abs(w) < epsT * erretm
            λ = dorg + τ
            return λ, true
        end

        if w <= 0
            dltlb = max(dltlb, τ)
        else
            dltbu = min(dltub, τ)
        end

        niter += 1
        η = next_step(ii, niter, w, τ, parts, switch3, origin_at_i, false)
        prev_w = w
        δ .= δ .- η
        w, erretm, parts = update_w(ii, τ, η)
        switch = false
        if origin_at_i
            if -w > abs(prev_w) / 10
                switch = true
            end
        else
            if w > abs(prev_w) / 10
                switch = true
            end
        end
        τ += η
        iter = niter + 1
        for niter in iter:maxit
            # test for convergence
            if abs(w) < epsT * erretm
                λ = dorg + τ
                return λ, true
            end

            if w <= 0
                dltlb = max(dltlb, τ)
            else
                dltbu = min(dltub, τ)
            end
            η = next_step(ii, niter, w, τ, parts, switch3, origin_at_i, switch)
            δ .= δ .- η
            τ += η
            prev_w = w
            w, erretm, parts = update_w(ii, τ, zero(T))
            if (w * prev_w > 0) && (abs(w) > abs(prev_w) / 10)
                switch = !switch
            end
        end
        # if we get here, convergence failed
        λ = dorg + τ
        return λ, false
    end
end

# root of secular equation for n=2
# based on DLAED5
function _dcroot2!(i,d::AbstractVector{T},z,δ,ρ) where {T}
    del = d[2] - d[1]
    if i == 1
        w = one(T) + 2 * ρ * (z[2]^2 - z[1]^2) / del
        if w > 0
            b = del + ρ * (z[1]^2 + z[2]^2)
            c = ρ * z[1]^2 * del
            τ = 2 * c / (b + sqrt(b^2 - 4 * c))
            λ = d[1] + τ
            δ[1] = -z[1] / τ
            δ[2] = z[2] / (del - τ)
        else
            b = -del  + ρ * (z[1]^2 + z[2]^2)
            c = ρ * z[2]^2 * del
            if b > 0
                τ = -2 * c / (b + sqrt(b^2 + 4 * c))
            else
                τ = (b - sqrt(b^2 + 4 * c)) / 2
            end
            λ = d[2] + τ
            δ[1] = -z[1] / (del + τ)
            δ[2] = -z[2] / τ
        end
        t = sqrt(δ[1]^2 + δ[2]^2)
        δ[1] = δ[1] / t
        δ[2] = δ[2] / t
    else
        # i == 2
        b = -del + ρ * (z[1]^2 + z[2]^2)
        c = ρ * z[2]^2 * del
        if b > 0
            τ = (b + sqrt(b^2 + 4*c)) / 2
        else
            τ = 2 * c / (-b + sqrt(b^2 + 4 * c))
        end
        λ = d[2] + τ
        δ[1] = -z[1] / (del + τ)
        δ[2] = -z[2] / τ
        t = sqrt(δ[1]^2 + δ[2]^2)
        δ[1] = δ[1] / t
        δ[2] = δ[2] / t
    end
    return λ
end

# Newton iteration for step in solution of secular equation
# based on DLAED6
function _dcnewton(kniter, origin_at_i, ρ, d::AbstractVector{T}, z, f_init) where {T}
    maxit = 40
    if origin_at_i
        lbd = d[2]
        ubd = d[3]
    else
        lbd = d[1]
        ubd = d[2]
    end
    if f_init < 0
        lbd = zero(T)
    else
        ubd = zero(T)
    end
    niter = 1
    τ = zero(T)
    if kniter == 2
        if origin_at_i
            t = (d[3] - d[2]) / 2
            c = ρ + z[1] / ((d[1] - d[2]) - t)
            a = c * (d[2] + d[3]) + z[2] + z[3]
            b = c * d[2] * d[3] + z[2] * d[3] + z[3] * d[2]
        else
            t = (d[1] - d[2]) / 2
            c = ρ + z[3] / ((d[3] - d[2]) - t)
            a = c * (d[1] + d[2]) + z[1] + z[2]
            b = c * d[1] * d[2] + z[1] * d[2] + z[2] * d[1]
        end
        t = max(abs(a), abs(b), abs(c))
        a /= t
        b /= t
        c /= t
        if c == 0
            τ = b/a
        elseif a < 0
            τ = (a - sqrt(abs(a^2 - 4 * b * c)))
        else
            τ = 2 * b / (a + sqrt(abs(a^2 - 4 * b * c)))
        end
        if τ < lbd || τ > ubd
            τ = (lbd + ubd) / 2
        end
        if d[1] == τ || d[2] == τ || d[3] == τ
            τ = zero(T)
        else
            t = (f_init
                 + τ * z[1] / (d[1] * (d[1] - τ))
                 + τ * z[2] / (d[2] * (d[2] - τ))
                 + τ * z[3] / (d[3] * (d[3] - τ))
            )
            if t <= 0
                lbd = τ
            else
                ubd = τ
            end
            if abs(f_init) <= abs(t)
                τ = zero(T)
            end
        end
    end # kniter == 2

    epsT = eps(T)
    small1 = T(2)^(Int(round(log2(floatmin(T)) / 3)))
    sminv1 = 1 / small1
    small2 = small1^2
    sminv2 = sminv1^2
    if origin_at_i
        t = min(abs(d[2] - τ), abs(d[3] - τ))
    else
        t = min(abs(d[1] - τ), abs(d[2] - τ))
    end
    scale = t <= small1
    if scale
        if t <= small2
            sclfac = sminv2
            sclinv = small2
        else
            sclfac = sminv1
            sclinv = small2
        end
        dscale = d * sclfac
        zscale = z * sclfac
        τ *= sclfac
        lbd *= sclfac
        ubd *= sclfac
    else
        dscale = d
        zscale = z
    end
    fc = zero(T)
    df = zero(T)
    ddf = zero(T)
    for i in 1:3
        t = one(T) / (dscale[1] - τ)
        t1 = zscale[i] * t
        t2 = t1 * t
        t3 = t2 * t
        fc += t1 / dscale[i]
        df += t2
        ddf += t3
    end
    f = f_init + τ * fc
    if abs(f) <= 0
        if scale
            τ *= sclinv
        end
        return τ, true
    end
    if f <= 0
        lbd = τ
    else
        ubd = τ
    end

    # Gragg-Thornton-Warner cubic convergent iteration
    iter = niter + 1
    converged = false
    for niter in iter:maxit
        if origin_at_i
            t1 = dscale[2] - τ
            t2 = dscale[3] - τ
        else
            t1 = dscale[1] - τ
            t2 = dscale[2] - τ
        end
        a = (t1 + t2) * f - t1 * t2 * df
        b = t1 * t2 * f
        c = f - (t1 + t2) * df + t1 * t2 * ddf
        t = max(abs(a), abs(b), abs(c))
        a /= t
        b /= t
        c /= t
        if c == 0
            η = b / a
        elseif a <= 0
            η = (a - sqrt(abs(a^2 - 4 * b * c))) / (2 * c)
        else
            η = 2 * b / (a + sqrt(abs(a^2 - 4 * b * c)))
        end
        if f * η >= 0
            η = -f / df
        end
        τ += η
        if τ < lbd || τ > ubd
            τ = (lbd + ubd) / 2
        end
        fc = zero(T)
        erretm = zero(T)
        df = zero(T)
        ddf = zero(T)
        for i in 1:3
            if (dscale[i] - τ) != 0
                t = 1 / (dscale[i] - τ)
                t1 = zscale[i] * t
                t2 = t1 * t
                t3 = t2 * t
                t4 = t1 / dscale[i]
                fc += t4
                erretm += abs(t4)
                df += t2
                ddf += t3
            else
                converged = true
                break
            end
        end
        if converged
            break
        end
        f = f_init + τ * fc
        erretm = 8 * (abs(f_init) + abs(τ) * erretm) + abs(τ) * df
        if (abs(f) <= 4 * epsT * erretm) || ((ubd - lbd) <= 4 * epsT * abs(τ))
            converged = true
            break
        end
        if f <= 0
            lbd = τ
        else
            ubd = τ
        end
    end

    if scale
        τ *= sclinv
    end
    return τ, converged
end
