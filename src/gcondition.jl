# select algorithm used to estimate Sep for triangular generalized problems
abstract type TrGSepAlgo end
# look-ahead is standard in LAPACK
struct TrGSepLookAhead <: TrGSepAlgo end
# alternative is Higham's iterative 1-norm estimator
struct TrGSepIter <: TrGSepAlgo end

function _trgsep(
        A::StridedMatrix{T}, B::StridedMatrix{T},
        D::StridedMatrix{T}, E::StridedMatrix{T},
        algo::TrGSepAlgo = TrGSepLookAhead()
    ) where {T}
    m = checksquare(A)
    n = checksquare(B)

    # C and F are used to hold intermediate results
    C = zeros(T, m, n)
    F = zeros(T, m, n)

    m1 = checksquare(D)
    n1 = checksquare(E)
    (m1 == m && n1 == n) || throw(
        DimensionMismatch(
            "dimensions of D and E must match A and B"
        )
    )

    rtyone = one(real(T))
    rtyzero = zero(real(T))
    scale = rtyone
    scaloc = rtyone
    dif = rtyzero
    cone = one(T)
    rdsum = rtyone
    rdscale = rtyzero
    for j in 1:n
        for i in m:-1:1
            # build 2x2 problem
            Z = [A[i, i] -B[j, j]; D[i, i] -E[j, j]]
            fZ = lu(Z) # should be _getc2
            rhs = [C[i, j]; F[i, j]]
            if algo isa TrGSepLookAhead
                # from zlatdf
                # set up a RHS b s.t. solution x = Z \ b has large norm
                x = rhs[fZ.p] # ipiv if _getc2
                pmone = -one(T)
                bp = x[1] + cone
                bm = x[1] - cone
                # L-part
                splus = rtyone
                L21 = fZ.L[2, 1]
                splus += real(L21' * L21)
                sminu = real(L21' * x[2])
                splus *= real(x[1])
                if splus > sminu
                    x[1] = bp
                elseif sminu > splus
                    x[1] = bm
                else
                    x[1] += pmone
                    pmone = cone
                end
                x[2] -= x[1] * L21
                # U-part
                w = [x[1]; x[2] + cone]
                x[2] -= cone
                splus = zero(real(T))
                sminu = zero(real(T))
                for i in 2:-1:1
                    t = cone / fZ.U[i, i]
                    w[i] *= t
                    x[i] *= t
                    if i == 1
                        k = 2
                        U12 = fZ.U[i, k]
                        w[i] -= w[k] * (U12 * t)
                        x[i] -= x[k] * (U12 * t)
                    end
                    splus += abs(w[i])
                    sminu += abs(x[i])
                end
                if splus > sminu
                    x .= w
                end
                x = x[fZ.p] # jpiv if _getc2
                rdscale, rdsum = _ssq(x, rdscale, rdsum)
                if rdscale != 0
                    dif = sqrt(2 * rtyone * m * n) / (rdscale * sqrt(rdsum))
                end
            else
                error("missing logic for $algo")
            end

            C[i, j] = x[1]
            F[i, j] = x[2]
            # substitute R[i,j], L[i,j] into remaining eq
            if i > 1
                α = -x[1]
                C[1:(i - 1), j] .+= α * A[1:(i - 1), i]
                F[1:(i - 1), j] .+= α * D[1:(i - 1), i]
            end
            if j < n
                α = x[2]
                C[i, (j + 1):n] .+= α * B[j, (j + 1):n]
                F[i, (j + 1):n] .+= α * E[j, (j + 1):n]
            end
        end
    end
    return dif
end

# based on ztgsen
"""
    eigvalscond(S::GeneralizedSchur, nsub) => pl, pr

compute approx. reciprocal norms of projectors on left/right subspaces
associated w/ leading `nsub`×`nsub` block of `S`. Use `ordschur` to select
eigenvalues of interest.

An approximate bound on avg. absolute error of associated eigenvalues
is `ϵ * norm(vcat(A,B)) / pl`. See LAPACK documentation for further details.
"""
function eigvalscond(S::GeneralizedSchur{Ty}, nsub) where {Ty}
    # nsub is m in ztgsen
    n = size(S.T, 1)
    rtyone = one(real(Ty))
    rtyzero = zero(real(Ty))
    if nsub == 0 || nsub == n
        return rtyone, rtyone
    end
    # solve S₁₁ R - L S₂₂ = σ S₁₂
    #       T₁₁ R - L T₂₂ = σ T₁₂
    n1 = nsub
    n2 = n - nsub
    R = S.S[1:n1, (n1 + 1):n]
    L = S.T[1:n1, (n1 + 1):n]
    R, L, scale = trsylvester!(
        S.S[1:n1, 1:n1], S.S[(n1 + 1):n, (n1 + 1):n], R,
        S.T[1:n1, 1:n1], S.T[(n1 + 1):n, (n1 + 1):n], L
    )

    #=
    ## DEVNOTE: this has been taking MINUTES to compile on v1.1.1+
    ## so we live with the allocations for now

    R, L, scale = trsylvester!(view(S.S,1:n1,1:n1),view(S.S,n1+1:n,n1+1:n),R,
                               view(S.T,1:n1,1:n1),view(S.T,n1+1:n,n1+1:n),L)
    =#

    rdscale = rtyzero
    sumsq = rtyone
    rdscale, sumsq = _ssq(vec(R), rdscale, sumsq)
    pl = rdscale * sqrt(sumsq)
    if pl == 0
        pl = rtyone
    else
        pl = scale / (sqrt(scale * scale / pl + pl) * sqrt(pl))
    end
    rdscale = rtyzero
    sumsq = rtyone
    rdscale, sumsq = _ssq(vec(L), rdscale, sumsq)
    pr = rdscale * sqrt(sumsq)
    if pr == 0
        pr = rtyone
    else
        pr = scale / (sqrt(scale * scale / pr + pr) * sqrt(pr))
    end
    return pl, pr
end

function subspacesep(
        S::GeneralizedSchur{Ty}, nsub::Integer;
        pnorm::Union{Int, Symbol} = :F
    ) where {Ty}
    n = size(S.S, 1)
    rtyone = one(real(Ty))
    n2 = n - nsub
    sep = zeros(real(Ty), 2)
    # these could use views for all, but for now that takes minutes to compile
    function fu(X)
        # solve T₁₁ R - R T₂₂ = σ X
        R = reshape(X[1:(nsub * n2)], nsub, n2)
        L = reshape(X[(nsub * n2 + 1):(2 * nsub * n2)], nsub, n2)
        R, L, s = trsylvester!(
            S.S[1:nsub, 1:nsub],
            S.S[(nsub + 1):n, (nsub + 1):n],
            R,
            S.T[1:nsub, 1:nsub],
            S.T[(nsub + 1):n, (nsub + 1):n],
            L
        )
        scale = s
        return vcat(vec(R), vec(L))
    end
    function fuH(X)
        # solve T₁₁ᴴ R - R T₂₂ᴴ = σ X
        R = reshape(X[1:(nsub * n2)], nsub, n2)
        L = reshape(X[(nsub * n2 + 1):(2 * nsub * n2)], nsub, n2)
        R, L, s = adjtrsylvester!(
            S.S[1:nsub, 1:nsub],
            S.S[(nsub + 1):n, (nsub + 1):n],
            R,
            S.T[1:nsub, 1:nsub],
            S.T[(nsub + 1):n, (nsub + 1):n],
            L
        )
        scale = s
        return vcat(vec(R), vec(L))
    end
    function fl(X)
        # solve T₁₁ R - R T₂₂ = σ X
        R = reshape(X[1:(nsub * n2)], n2, nsub)
        L = reshape(X[(nsub * n2 + 1):(2 * nsub * n2)], n2, nsub)
        R, L, s = trsylvester!(
            S.S[(nsub + 1):n, (nsub + 1):n],
            S.S[1:nsub, 1:nsub],
            R,
            S.T[(nsub + 1):n, (nsub + 1):n],
            S.T[1:nsub, 1:nsub],
            L
        )
        scale = s
        return vcat(vec(R), vec(L))
    end
    function flH(X)
        # solve T₁₁ᴴ R - R T₂₂ᴴ = σ X
        R = reshape(X[1:(nsub * n2)], n2, nsub)
        L = reshape(X[(nsub * n2 + 1):(2 * nsub * n2)], n2, nsub)
        R, L, s = adjtrsylvester!(
            S.S[(nsub + 1):n, (nsub + 1):n],
            S.S[1:nsub, 1:nsub],
            R,
            S.T[(nsub + 1):n, (nsub + 1):n],
            S.T[1:nsub, 1:nsub],
            L
        )
        scale = s
        return vcat(vec(R), vec(L))
    end
    if pnorm == :F || pnorm == :Frobenius
        sep[1] = _trgsep(
            view(S.S, 1:nsub, 1:nsub),
            view(S.S, (nsub + 1):n, (nsub + 1):n),
            view(S.T, 1:nsub, 1:nsub),
            view(S.T, (nsub + 1):n, (nsub + 1):n)
        )
        sep[2] = _trgsep(
            view(S.S, (nsub + 1):n, (nsub + 1):n),
            view(S.S, 1:nsub, 1:nsub),
            view(S.T, (nsub + 1):n, (nsub + 1):n),
            view(S.T, 1:nsub, 1:nsub)
        )
    elseif pnorm == 1
        scale = rtyone
        estu = norm1est!(fu, fuH, zeros(Ty, 2 * nsub * n2))
        sep[1] = scale / estu
        scale = rtyone
        estl = norm1est!(fl, flH, zeros(Ty, 2 * nsub * n2))
        sep[2] = scale / estl
    else
        throw(ArgumentError("pnorm must be :Frobenius or 1"))
    end
    return sep
end
