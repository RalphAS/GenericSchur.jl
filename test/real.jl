@info "Suppressing 2x2 block checks pending standardization logic."

# this requires that tiny values have been cleaned out
function checkblocks(A::StridedMatrix; debug=false)
    n = size(A,1)
    isok = true
    for j=1:n-1
        if A[j+1,j] != 0
            isok &= (A[j,j] == A[j+1,j+1])
            isok &= (A[j,j+1] != 0)
            isok &= (A[j,j+1] * A[j+1,j] < 0) # signs must differ (overflow?)
        end
        if debug && !isok
            print("problem block: ",A[j:j+1,j:j+1])
            (A[j,j] == A[j+1,j+1]) || print(" diag")
            (A[j,j+1] != 0) || print(" 0_super")
            (A[j,j+1] * A[j+1,j] < 0) || print(" trail_sign")
            println()
            return false
        end
    end
    isok
end

function schurtest(A::Matrix{T}, tol; normal=false) where {T<:Real}
    n = size(A,1)
    ulp = eps(T)
    if T <: BlasFloat
        S = GenericSchur.gschur(A)
    else
        S = schur(A)
    end
    # test 1: S.T is upper quasi-triangular
    @test all(tril(S.T,-2) .== 0)
    # FIXME: suppress pending cleaning logic
    # check complex 2x2 blocks
    # @test checkblocks(S.T, debug=true)
    # test 2: norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp) < tol
    @test norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp) < tol
    # test 3: S.Z is orthogonal: norm(I - S.Z * S.Z') / (n * ulp) < tol
    @test norm(I - S.Z * S.Z') / (n * ulp) < tol
    # test 4: S.values are e.v. of T
    if T <: BlasFloat
        vLA = csort(eigvals(S.T))
        vGS = csort(S.values)
        rte = sqrt(eps(T))
        @test isapprox(vGS, vLA, atol=rte, rtol=rte)
    end

    # It is tempting to check eigenvalues against LAPACK eigvals(A)
    # for suitable types, but the comparison is misleading without
    # detailed analysis because many of the test matrices are severly
    # non-normal.

    if normal
        # verify that Schur diagonalizes normal matrices
        # necessary but not really sufficient (complex blocks)
        @test norm(triu(S.T,2)) / (n * norm(A) * ulp) < tol
    end
end

# random orthogonal matrix
function rando(::Type{T},n) where {T<:Real}
    if T ∈ [Float16,Float32,Float64]
        A = randn(T,n,n)
        F = qr(A)
        return Matrix(F.Q)
    else
        # don't have normal deviates for other types, but need appropriate QR
        A = randn(Float64,n,n)
        F = qr(convert.(T,A))
        return Matrix(F.Q)
    end
end

Random.seed!(1234)

for T in [BigFloat,Float16]
    @testset "group $T" begin
        unfl = floatmin(T)
        ovfl = one(T) / unfl
        ulp = eps(T)
        ulpinv = one(T) / ulp
        rtulp = sqrt(ulp)
        rtulpi = one(T) / rtulp

        ens = [4,32]
        @testset "random general" begin
            for n in ens
                A = rand(T,n,n)
                schurtest(A,10)
            end
        end
        @testset "random orthogonal" begin
            for n in ens
                A = rando(T,n)
                schurtest(A,10,normal=true)
            end
        end

    end # group testset
end # type loop (non-BlasFloat)

@testset "Godunov" begin
    setprecision(BigFloat, 80) do
        A,v = godunov(BigFloat)
        S = schur(A)
        @test isapprox(csort(S.values),v,atol=2e-5)
        vnew = eigvals(A)
        @test isapprox(csort(vnew),v,atol=2e-5)
    end
end

for T in [Float64, Float32]
    @testset "group $T" begin

        unfl = floatmin(T)
        ovfl = one(T) / unfl
        ulp = eps(T)
        ulpinv = one(T) / ulp
        rtulp = sqrt(ulp)
        rtulpi = one(T) / rtulp

        # sizes of matrices to test
        ens = [4,32]

        itypemax = 10
        # On author's dev system, tests pass with tol=5 for most and 6 for all,
        # but generalization is risky.
        # (LAPACK uses 20, FWIW.)
        tols = fill(20,itypemax)

        simrconds = [1.0, rtulpi, 0.0]
        magns = [1.0, ovfl*ulp, unfl*ulpinv]

        @testset "Jordan" begin
            itype=3
            kmagn = [1]
            nj = length(kmagn)
            for n in ens
                for j in 1:nj
                    anorm = magns[kmagn[j]]
                    A = diagm(1 => ones(T,n-1)) + anorm * I
                    schurtest(A,tols[itype])
                end
            end
        end

        @testset "general, ev specified" begin
            itype=6
            kmagn =  [1,1,1,1,1,1,1,1,2,3]
            kmode =  [4,3,1,5,4,3,1,5,5,5]
            kconds = [1,1,1,1,2,2,2,2,2,2]
            nj = length(kmode)
            for n in ens
                for j in 1:nj
#               CALL DLATME( N, 'S', ISEED, WORK, IMODE, COND, CONE,
#     $                      ' ', 'T', 'T', 'T', RWORK, 4, CONDS, N, N, ANORM,
#     $                      A, LDA, WORK( 2*N+1 ), IINFO )
                    A = zeros(T,n,n)
                    simrcond = simrconds[kconds[j]]
                    imode = kmode[j]
                    anorm = magns[kmagn[j]]
                    rcond = ulpinv
                    verbosity[] > 1 &&
                        println("latme: n=$n $anorm, $imode, $rcond, $simrcond")
                    latme!(A,anorm,imode,rcond,simrcond)
                    schurtest(A,tols[itype])
                end
            end
        end

        @testset "diagonal, random ev" begin
            itype=7
            kmode = [6]
            kmagn = [1]
            nj = length(kmagn)
            for n in ens
                for j in 1:nj
     #           CALL ZLATMR( N, N, 'D', ISEED, 'N', WORK, 6, ONE, CONE,
     # $                      'T', 'N', WORK( N+1 ), 1, ONE,
     # $                      WORK( 2*N+1 ), 1, ONE, 'N', IDUMMA, 0, 0,
     # $                      ZERO, ANORM, 'NO', A, LDA, IWORK, IINFO )
                    A = zeros(T,n,n)
                    imode = kmode[j]
                    anorm = magns[kmagn[j]]
                    rcond = 1.0
                    latmr!(A,anorm,imode,rcond,kl=0,ku=0)
                    schurtest(A,tols[itype])
                end
            end
        end

        @testset "symmetric, random ev" begin
            itype=8
            kmagn = [1]
            kmode = [6]
            nj = length(kmagn)
            for n in ens
                for j in 1:nj
     # note sym='H'
     #           CALL ZLATMR( N, N, 'D', ISEED, 'H', WORK, 6, ONE, CONE,
     # $                      'T', 'N', WORK( N+1 ), 1, ONE,
     # $                      WORK( 2*N+1 ), 1, ONE, 'N', IDUMMA, N, N,
     # $                      ZERO, ANORM, 'NO', A, LDA, IWORK, IINFO )
                    A = zeros(T,n,n)
                    imode = kmode[j]
                    anorm = magns[kmagn[j]]
                    rcond = 1.0
                    latmr!(A,anorm,imode,rcond,sym='H',kl=n,ku=n)
                    schurtest(A,tols[itype])
                end
            end
        end

        @testset "general, random ev" begin
            itype=9
            kmagn = [1,2,3]
            # kmode = [4,3,1] # data statement is a lie
            kmode = [6,6,6]
            nj = length(kmagn)
            for n in ens
                for j in 1:nj
     #           CALL ZLATMR( N, N, 'D', ISEED, 'N', WORK, 6, ONE, CONE,
     # $                      'T', 'N', WORK( N+1 ), 1, ONE,
     # $                      WORK( 2*N+1 ), 1, ONE, 'N', IDUMMA, N, N,
     # $                      ZERO, ANORM, 'NO', A, LDA, IWORK, IINFO )
            # zdrves zeros first 2 rows/cols & last row w/o explanation
            # presumably to test balancing, which we don't do, so skip it.
                    A = zeros(T,n,n)
                    imode = kmode[j]
                    anorm = magns[kmagn[j]]
                    rcond = 1.0
                    (verbosity[] > 1) &&
                        println("latmr: n=$n $anorm, $imode, $rcond")
                    latmr!(A,anorm,imode,rcond)
                    schurtest(A,tols[itype])
                end
            end
        end

        @testset "triangular, random ev" begin
            # trivial, not even run in zdrves
            itype=10
            kconds = [1]
            kmagn = [1]
            kmode = [1]
            nj = length(kmagn)
            for n in ens
                for j in 1:nj
     # note: kl=0
     #           CALL ZLATMR( N, N, 'D', ISEED, 'N', WORK, 6, ONE, CONE,
     # $                      'T', 'N', WORK( N+1 ), 1, ONE,
     # $                      WORK( 2*N+1 ), 1, ONE, 'N', IDUMMA, N, 0,
     # $                      ZERO, ANORM, 'NO', A, LDA, IWORK, IINFO )
#
                    A = zeros(T,n,n)
                    imode = 6
                    anorm = magns[kmagn[j]]
                    rcond = 1.0
                    latmr!(A,anorm,imode,rcond,kl=0)
                    schurtest(A,tols[itype])
                end
            end
        end


    end # group testset
end # type loop (BlasFloat)
