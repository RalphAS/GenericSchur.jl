
function checkord(A::Matrix{Ty}, tol=10) where {Ty<:Complex}
    n = size(A,1)
    ulp = eps(real(Ty))

    S = schur(A)
    T2 = copy(S.T)
    Z2 = copy(S.Z)
    select = fill(true,n)
    for i=1:(n>>1)
        select[i] = false
    end
    T2,Z2,v = GenericSchur.gordschur!(T2,Z2,select)

    # usual tests for Schur
    @test all(tril(T2,-1) .== 0)
    @test norm(Z2*T2*Z2' - A) / (n * norm(A) * ulp) < tol
    @test norm(I - Z2 * Z2') / (n * ulp) < tol

    # make sure we got the ones we asked for
    nwanted = count(select)
    wwanted = [S.values[j] for j=1:n if select[j]]
    errs = zeros(nwanted)
    for i=1:nwanted
        w = T2[i,i]
        errs[i] = minimum(abs.(wwanted .- w)) / (ulp + abs(w))
    end
    @test all(errs .< tol)
end

using LinearAlgebra.LAPACK: trsen!, BlasInt

function checkecond(A::Matrix{T}, nsub=3) where T
    n = size(A,1)
    select = fill(false,n)
    inds = rand(1:n,nsub)
    for i in inds
        select[i] = true
    end
    S = schur(A)

    @assert real(T) <: BlasFloat
    T1 = copy(S.T)
    Z1 = copy(S.Z)
    fselect = BlasInt.([x ? 1 : 0 for x in select])
    T1,Z1,w1,s1,sep1 = trsen!('B','V',fselect,T1,Z1)

    S2 = ordschur(S, select)
    s2 = eigvalscond(S2, count(select))
    sep2 = subspacesep(S2, count(select))

    @test s1 ≈ s2
    @test sep1 ≈ sep2

end

for Ty in [Complex{Float64}]
    @testset "ordschur $Ty" begin
        n = 32
        A = rand(Ty,n,n)
        checkord(A)
    end
    @testset "subspace condition $Ty" begin
        n = 32
        A = rand(Ty,n,n)
        checkecond(A)
    end
end

function checkord(A::Matrix{Ty}, tol=10) where {Ty<:Real}
    n = size(A,1)
    ulp = eps(real(Ty))

    S = schur(A)
    v = S.values
    # cases to try: 2 reals, 1 complex pair, 1 real & 1 pair, 2 pairs
    nr = [2,0,1,0]
    ni = [0,2,2,4]
    for icase in 1:4
        T2 = copy(S.T)
        Z2 = copy(S.Z)
        select = fill(false,n)
        i = n+1
        nr_needed = nr[icase]
        ni_needed = ni[icase]
        while (nr_needed + ni_needed > 0) && i>2
            i -= 1
            if nr_needed > 0
                if imag(v[i]) == 0
                    select[i] = true
                    nr_needed -= 1
                end
            end
            if ni_needed > 0
                ivi = imag(v[i])
                if (abs(ivi) > 1000ulp) && (imag(v[i-1]) == -ivi)
                    select[i] = true
                    select[i-1] = true
                    ni_needed -= 2
                    i -= 1
                end
            end
        end
        if nr_needed + ni_needed > 0
            @warn "failed to find acceptable subset (class $icase) for checking ordschur"
            continue
        end
        T2,Z2,v = GenericSchur.gordschur!(T2,Z2,select)

        # usual tests for Schur
        @test all(tril(T2,-2) .== 0)
        @test norm(Z2*T2*Z2' - A) / (n * norm(A) * ulp) < tol
        @test norm(I - Z2 * Z2') / (n * ulp) < tol

        # make sure we got the ones we asked for
        nwanted = count(select)
        wwanted = [S.values[j] for j=1:n if select[j]]
        errs = zeros(nwanted)
        for i=1:nwanted
            w = T2[i,i]
            errs[i] = minimum(abs.(wwanted .- w)) / (ulp + abs(w))
        end
        @test all(errs .< tol)
    end
end

for Ty in [Float64, BigFloat]
    @testset "ordschur $Ty" begin
        # should be big enough to be nearly certain of meaningful selection
        n = 20
        A = rand(Ty,n,n)
        checkord(A)
    end
end
