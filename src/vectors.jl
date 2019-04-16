# Compute right eigenvectors of a complex upper triangular matrix TT.
# If Z is nontrivial, multiply by it to get eigenvectors of Z*TT*Z'.
# based on LAPACK::ztrevc
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
function _geigvecs!(TT::StridedMatrix{T},
                    Z::StridedMatrix{T}=Matrix{T}(undef,0,0)
                    ) where {T <: Complex}
    n = size(TT,1)
    RT = real(T)
    ulp = eps(RT)
    smallnum = safemin(RT) * (n / ulp)
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
            vscale = _usolve!(TT,ki-1,v,tnorms)
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

# based on LAPACK::ztrevc
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
function _gleigvecs!(TT::StridedMatrix{T},
                    Z::StridedMatrix{T}=Matrix{T}(undef,0,0)
                    ) where {T <: Complex}
    n = size(TT,1)
    RT = real(T)
    ulp = eps(RT)
    smallnum = safemin(RT) * (n / ulp)
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
