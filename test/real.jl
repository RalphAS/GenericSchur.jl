# Check that 2x2 diagonal blocks have standard form.
# This requires that tiny values have been cleaned out.
function checkblocks(A::StridedMatrix; debug = false)
    n = size(A, 1)
    isok = true
    for j in 1:(n - 1)
        if A[j + 1, j] != 0
            isok &= (A[j, j] == A[j + 1, j + 1])
            isok &= (A[j, j + 1] != 0)
            isok &= (sign(A[j, j + 1]) * sign(A[j + 1, j]) < 0) # signs must differ (overflow?)
        end
        if debug && !isok
            print("problem block: ", A[j:(j + 1), j:(j + 1)])
            (A[j, j] == A[j + 1, j + 1]) || print(" diag")
            (A[j, j + 1] != 0) || print(" 0_super")
            (sign(A[j, j + 1]) * sign(A[j + 1, j]) < 0) || print(" trail_sign")
            println()
            return false
        end
    end
    return isok
end

# This only applies to standard forms w/ co-sorted eigenvals
function checkeigvals(
        A::StridedMatrix{T}, w::StridedVector, tol; debug = false
    ) where {T <: Real}
    n = size(A, 1)
    ulp = eps(T)
    tiny = GenericSchur.safemin(T)
    isok = true
    for j in 1:n
        isok &= (A[j, j] == real(w[j]))
    end
    if n > 1
        if A[2, 1] == 0
            isok &= (imag(w[1]) == 0)
        end
        if A[n, n - 1] == 0
            isok &= (imag(w[n]) == 0)
        end
    end
    for j in 1:(n - 1)
        if A[j + 1, j] != 0
            t = sqrt(abs(A[j + 1, j])) * sqrt(abs(A[j, j + 1]))
            cmp = max(ulp * t, tiny)
            isok &= (abs(imag(w[j]) - t) / cmp < tol)
            isok &= (abs(imag(w[j + 1]) + t) / cmp < tol)
        elseif (j > 1) && (A[j + 1, j] == 0) && (A[j, j - 1] == 0)
            isok &= (imag(w[j]) == 0)
        end
    end
    return isok
end

function schurtest(
        A::Matrix{T}, tol; normal = false, baddec = false,
    ) where {T <: Real}
    n = size(A, 1)
    ulp = eps(T)
    # WARNING: if using wrappers, schur() uses eigtype(),
    # which promotes ComplexF16 to ComplexF32
    S = GenericSchur.gschur(A)
    # test 1: S.T is upper quasi-triangular
    @test all(tril(S.T, -2) .== 0)
    # check complex 2x2 blocks
    @test checkblocks(S.T)
    # test 2: norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp) < tol
    if baddec
        @test_broken norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp) < tol
    else
        @test norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp) < tol
    end
    # test 3: S.Z is orthogonal: norm(I - S.Z * S.Z') / (n * ulp) < tol
    @test norm(I - S.Z * S.Z') / (n * ulp) < tol
    # test 4: S.values are e.v. of T
    @test checkeigvals(S.T, S.values, tol)

    # It is tempting to check eigenvalues against LAPACK eigvals(A)
    # for suitable types, but the comparison is misleading without
    # detailed analysis because many of the test matrices are severly
    # non-normal.

    if normal
        # verify that Schur "diagonalizes" normal matrices
        # necessary but not really sufficient (complex blocks)
        @test norm(triu(S.T, 2)) / (n * norm(A) * ulp) < tol
        # TODO: in this case comparison to LAPACK eigenvalues is legit
    end

    Sc = triangularize(S)

    # test 1: S.T is upper triangular
    @test all(tril(Sc.T, -1) .== 0)
    # test 2: norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp) < tol
    @test norm(A - Sc.Z * Sc.T * Sc.Z') / (n * norm(A) * ulp) < tol
    # test 3: S.Z is unitary: norm(I - S.Z * S.Z') / (n * ulp) < tol
    @test norm(I - Sc.Z * Sc.Z') / (n * ulp) < tol
    # test 4: S.values are e.v. of T
    @test all(csort(Sc.values) .== csort(diag(Sc.T)))
    return
end

function hesstest(A::Matrix{T}, tol) where {T <: Real}
    n = size(A, 1)
    ulp = eps(real(T))
    H = invoke(
        GenericSchur._hessenberg!,
        Tuple{StridedMatrix{Ty}} where {Ty},
        copy(A)
    )
    Q = GenericSchur._materializeQ(H)
    if VERSION < v"1.3"
        HH = triu(H.data, -1)
    else
        HH = H.H
        # test 1: H.H is upper triangular
        @test all(tril(HH, -2) .== 0)
    end
    # test 2: norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp) < tol
    decomp_err = norm(A - Q * HH * Q') / (n * norm(A) * ulp)
    @test decomp_err < tol
    # test 3: S.Z is orthogonal: norm(I - S.Z * S.Z') / (n * ulp) < tol
    orth_err = norm(I - Q * Q') / (n * ulp)
    @test orth_err < tol
    return
end

# random orthogonal matrix
function rando(::Type{T}, n) where {T <: Real}
    if T ∈ [Float16, Float32, Float64]
        A = randn(T, n, n)
        F = qr(A)
        return Matrix(F.Q)
    else
        # don't have normal deviates for other types, but need appropriate QR
        A = randn(Float64, n, n)
        F = qr(convert.(T, A))
        return Matrix(F.Q)
    end
end

Random.seed!(1234)

# Don't try BigFloat here because its floatmin is perverse.
@testset "Hessenberg $T" for T in [Float64, Float16]
    n = 32
    tol = 10
    A = rand(T, n, n)
    hesstest(A, tol)
    # exercise some underflow-related logic
    s = 100 * floatmin(real(T))
    Asmall = s * A
    hesstest(Asmall, tol)
    s = floatmax(real(T)) / 100
    Abig = s * A
    hesstest(Abig, tol)
end

@testset "group $T" for T in [BigFloat, Float16]
    unfl = floatmin(T)
    ovfl = one(T) / unfl
    ulp = eps(T)
    ulpinv = one(T) / ulp
    rtulp = sqrt(ulp)
    rtulpi = one(T) / rtulp

    ens = [4, 32]
    @testset "random general" begin
        for n in ens
            A = rand(T, n, n)
            schurtest(A, 10)
        end
    end
    @testset "random orthogonal" begin
        for n in ens
            A = rando(T, n)
            schurtest(A, 10, normal = true)
        end
    end

end # group testset

@testset "Godunov" begin
    # Set precision fine enough for eigval condition to apply,
    # but not fine enough to make it a slam-dunk.
    setprecision(BigFloat, 80) do
        A, v, econd = godunov(BigFloat)
        S = GenericSchur.gschur(A)
        δ = norm(S.Z * S.T * S.Z' - A)
        @test δ < 100 * eps(big(1.0)) * norm(A)
        t = 3 * δ * econd
        @test isapprox(csort(S.values), v, atol = t)
        if piracy
            vnew = eigvals(A)
            @test isapprox(csort(vnew), v, atol = t)
        end
    end
end

@testset "group $T" for T in [Float64, Float32]

    unfl = floatmin(T)
    ovfl = one(T) / unfl
    ulp = eps(T)
    ulpinv = one(T) / ulp
    rtulp = sqrt(ulp)
    rtulpi = one(T) / rtulp

    # sizes of matrices to test
    ens = [4, 32]

    itypemax = 10
    # On author's dev system, tests pass with tol=5 for most and 6 for all,
    # but generalization is risky.
    # (LAPACK uses 20, FWIW.)
    tols = fill(20, itypemax)

    simrconds = [1.0, rtulpi, 0.0]
    magns = [1.0, ovfl * ulp, unfl * ulpinv]

    @testset "Jordan" begin
        itype = 3
        kmagn = [1]
        nj = length(kmagn)
        for n in ens
            for j in 1:nj
                anorm = magns[kmagn[j]]
                A = diagm(1 => ones(T, n - 1)) + anorm * I
                schurtest(A, tols[itype])
            end
        end
    end

    @testset "general, ev specified" begin
        itype = 6
        kmagn = [1, 1, 1, 1, 1, 1, 1, 1, 2, 3]
        kmode = [4, 3, 1, 5, 4, 3, 1, 5, 5, 5]
        kconds = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        nj = length(kmode)
        for n in ens
            for j in 1:nj
                #               CALL DLATME( N, 'S', ISEED, WORK, IMODE, COND, CONE,
                #     $                      ' ', 'T', 'T', 'T', RWORK, 4, CONDS, N, N, ANORM,
                #     $                      A, LDA, WORK( 2*N+1 ), IINFO )
                A = zeros(T, n, n)
                simrcond = simrconds[kconds[j]]
                imode = kmode[j]
                anorm = magns[kmagn[j]]
                rcond = ulpinv
                verbosity[] > 1 &&
                    println("latme: n=$n $anorm, $imode, $rcond, $simrcond")
                latme!(A, anorm, imode, rcond, simrcond)
                schurtest(A, tols[itype])
            end
        end
    end

    @testset "diagonal, random ev" begin
        itype = 7
        kmode = [6]
        kmagn = [1]
        nj = length(kmagn)
        for n in ens
            for j in 1:nj
                #           CALL ZLATMR( N, N, 'D', ISEED, 'N', WORK, 6, ONE, CONE,
                # $                      'T', 'N', WORK( N+1 ), 1, ONE,
                # $                      WORK( 2*N+1 ), 1, ONE, 'N', IDUMMA, 0, 0,
                # $                      ZERO, ANORM, 'NO', A, LDA, IWORK, IINFO )
                A = zeros(T, n, n)
                imode = kmode[j]
                anorm = magns[kmagn[j]]
                rcond = 1.0
                latmr!(A, anorm, imode, rcond, kl = 0, ku = 0)
                schurtest(A, tols[itype])
            end
        end
    end

    @testset "symmetric, random ev" begin
        itype = 8
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
                A = zeros(T, n, n)
                imode = kmode[j]
                anorm = magns[kmagn[j]]
                rcond = 1.0
                latmr!(A, anorm, imode, rcond, sym = 'H', kl = n, ku = n)
                schurtest(A, tols[itype])
            end
        end
    end

    @testset "general, random ev" begin
        itype = 9
        kmagn = [1, 2, 3]
        # kmode = [4,3,1] # data statement is a lie
        kmode = [6, 6, 6]
        nj = length(kmagn)
        for n in ens
            for j in 1:nj
                #           CALL ZLATMR( N, N, 'D', ISEED, 'N', WORK, 6, ONE, CONE,
                # $                      'T', 'N', WORK( N+1 ), 1, ONE,
                # $                      WORK( 2*N+1 ), 1, ONE, 'N', IDUMMA, N, N,
                # $                      ZERO, ANORM, 'NO', A, LDA, IWORK, IINFO )
                # zdrves zeros first 2 rows/cols & last row w/o explanation
                # presumably to test balancing, which we don't do, so skip it.
                A = zeros(T, n, n)
                imode = kmode[j]
                anorm = magns[kmagn[j]]
                rcond = 1.0
                (verbosity[] > 1) &&
                    println("latmr: n=$n $anorm, $imode, $rcond")
                latmr!(A, anorm, imode, rcond)
                schurtest(A, tols[itype])
            end
        end
    end

    @testset "triangular, random ev" begin
        # trivial, not even run in zdrves
        itype = 10
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
                A = zeros(T, n, n)
                imode = 6
                anorm = magns[kmagn[j]]
                rcond = 1.0
                latmr!(A, anorm, imode, rcond, kl = 0)
                schurtest(A, tols[itype])
            end
        end
    end

    @testset "tiny, almost degenerate" begin
        λ1, λ2 = (one(T) + 2eps(T), one(T) - 2eps(T))
        B = diagm(0 => [λ1, λ2]) + (eps(T) / 4) * rand(T, 2, 2)
        G, w1, w2 = GenericSchur._gs2x2!(B, 3)
        @test abs(max(w1, w2) - λ1) < 2eps(T)
        @test abs(min(w1, w2) - λ2) < 2eps(T)
    end
end # group testset
