# This file is part of GenericSchur.jl, released under the MIT "Expat" license

struct Balancer{T}
    ilo :: Int
    ihi :: Int
    prow :: Vector{Int}
    pcol :: Vector{Int}
    D :: Vector{T}
    trivial :: Bool
end

# intended to be a translation of xGEBAL from LAPACK
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
# w/ some allowance for scheme from R.James et al. 2014 arxiv 1401.5766
"""
    balance!(A; scale=true, permute=true) => Abal, B::Balancer

Balance a matrix so that various operations are more stable.

If `permute`, then row and column permutations are found such
that `Abal` has the block form `[T₁ X Y; 0 C Z; 0 0 T₂]` where
`T₁` and `T₂` are upper-triangular.
If `scale`, then a diagonal similarity transform using powers of 2
is found such that the 1-norm of `Abal` is near unity. The transformations
are encoded into `B` so that they can be inverted for eigenvectors, etc.
Balancing typically improves the accuracy of eigen-analysis.
"""
function balance!(A::AbstractMatrix{T}; scale=true, permute=true,
                  p::Int=1, algo = :pr) where T <: Number
    n = checksquare(A)
    RT = real(T)
    β = RT(2)
    factor = RT(0.95)

    sfmin1 = floatmin(RT) / eps(RT)
    sfmax1 = one(RT) / sfmin1
    sfmin2 = sfmin1 * β
    sfmax2 = one(RT) / sfmin2

    D = ones(T,n)
    ilo = 1
    ihi = n
    trivial = true
    sp = fill(0,n)

    if permute
        # look for row permutations which make the lower portion UT
        ihi = n+1
        js = 0
        @inbounds while ihi > 1
            ihi -= 1
            exch = false
            ms = 0
            for j=ihi:-1:1
                exch = true
                js = j
                for i=1:ihi
                    (i == j) && continue
                    if A[j,i] != 0
                        exch = false
                    end
                end
                if exch
                    ms = ihi
                    break
                end
            end
            if exch
                sp[ms] = js
                if js != ms
                    trivial = false
                    for i=1:ihi
                        A[i,js], A[i,ms] = A[i,ms], A[i,js]
                    end
                    for i=ilo:n
                        A[js,i], A[ms,i] = A[ms,i], A[js,i]
                    end
                end
            else
                break
            end
        end
        # look for column permutations which make the left portion UT
        ilo = 0
        @inbounds while ilo < n
            ilo += 1
            exch = false
            ms = 0
            js = 0
            for j=ilo:ihi
                js = j
                exch = true
                for i=ilo:ihi
                    (i == j) && continue
                    if A[i,j] != 0
                        exch = false
                    end
                end
                if exch
                    ms = ilo
                    break
                end
            end
            if exch
                sp[ms] = js
                if ms != js
                    trivial = false
                    for i=1:ihi
                        A[i,js], A[i,ms] = A[i,ms], A[i,js]
                    end
                    for i=ilo:n
                        A[js,i], A[ms,i] = A[ms,i], A[js,i]
                    end
                end
            else
                break
            end
        end
    end

    sf(c,r) = (p == 1) ? (c + r) : c*c + r*r
    # abs(c) + abs(r), but args always nonneg

    if scale
        converged = false
        while !converged
            converged = true
            for i=ilo:ihi
                if algo == :pr
                    # LAPACK mixes norms
                    c = norm(view(A,ilo:ihi,i),2)
                    r = norm(view(A,i,ilo:ihi),2)
                else
                    c = norm(view(A,ilo:ihi,i),p)
                    r = norm(view(A,i,ilo:ihi),p)
                end
                ca,_ = _findamax(view(A,ilo:ihi,i))
                ra,_ = _findamax(view(A,i,ilo:ihi))
                if c == 0 || r == 0
                    continue
                end
                g = r / β
                s = sf(c,r)
                f = one(RT)
                while c < r/β
                    if c >= g || (max(f,c,ca) >= sfmax2) || (min(r,g,ra) <= sfmin2)
                        break
                    end
                    if isnan(c+f+ca+r+g+ra)
                        error("NaN encountered while balancing")
                    end
                    f *= β
                    c *= β
                    ca *= β
                    r /= β
                    g /= β
                    ra /= β
                end
                g = c / β
                while r <= c/β
                    if (g < r) || (max(r,ra) >= sfmax2) || (min(f,c,g,ca) <= sfmin2)
                        break
                    end
                    f /= β
                    c /= β
                    g /= β
                    ca /= β
                    r *= β
                    ra *= β
                end
                if f != one(RT)
                    trivial = false
                end
                (sf(c,r) >= factor * s) && continue
                # TODO: continue on [uo]flow
                converged = false
                D[i] *= f
                A[i,ilo:n] .*= (one(T) / f)
                A[1:ihi,i] .*= f
            end
        end # while !converged
    end # if scale
    prow = sp[1:ilo-1]
    pcol = sp[ihi+1:n]
    B = Balancer(ilo,ihi,prow,pcol,D,trivial)
    return A, B
end

# this is the appropriate transformation for right eigenvectors
# stored as columns of a matrix
function LinearAlgebra.lmul!(B::Balancer, A::StridedMatrix{T}) where T
    n = checksquare(A)
    if B.trivial
        return A
    end
    ilo, ihi = B.ilo, B.ihi
    if ilo != ihi
        lmul!(Diagonal(B.D), A)
    end
    for j=ilo-1:-1:1
        m = B.prow[j]
        if m == j
            continue
        end
        for i=1:n
            A[j,i], A[m,i] = A[m,i], A[j,i]
        end
    end
    for j=ihi+1:n
        m = B.pcol[j-ihi]
        if m == j
            continue
        end
        for i=1:n
            A[j,i], A[m,i] = A[m,i], A[j,i]
        end
    end
    A
end

# this should be the appropriate transformation for left eigenvectors
# stored as columns of a matrix
function LinearAlgebra.ldiv!(B::Balancer, A::StridedMatrix{T}) where T
    n = checksquare(A)
    if B.trivial
        return A
    end
    ilo, ihi = B.ilo, B.ihi
    if ilo != ihi
        lmul!(Diagonal(one(T) ./ B.D), A)
    end
    for j=ilo-1:-1:1
        m = B.prow[j]
        if m == j
            continue
        end
        for i=1:n
            A[j,i], A[m,i] = A[m,i], A[j,i]
        end
    end
    for j=ihi+1:n
        m = B.pcol[j-ihi]
        if m == j
            continue
        end
        for i=1:n
            A[j,i], A[m,i] = A[m,i], A[j,i]
        end
    end
    A
end
