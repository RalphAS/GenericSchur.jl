using Printf

# isolate eigvec tests for use with special cases
function vectest(A::Matrix{T}, S::Schur{T2}, vtol; normal=false) where {T<:Complex, T2<:Complex}
    n = size(A,1)
    ulp = eps(real(T))
    VR = eigvecs(S)
    if verbosity[] > 1
        vec_err = norm(A * VR - VR * diagm(0 => S.values)) / (n * norm(A) * ulp)
        println("r.eigenvector error: $vec_err, vtol = $vtol")
        @test vec_err < 1000
    else
        @test norm(A * VR - VR * diagm(0 => S.values)) / (n * norm(A) * ulp) < vtol
    end
    VL = eigvecs(S, left=true)
    if verbosity[] > 1
        vec_err = norm(A' * VL - VL * diagm(0 => conj.(S.values))) / (n * norm(A) * ulp)
        println("l.eigenvector error: $vec_err, vtol = $vtol")
        @test vec_err < 1000
    else
        @test norm(A' * VL - VL * diagm(0 => conj.(S.values))) / (n * norm(A) * ulp) < vtol
    end
    if normal
        # check orthonormality of vectors where appropriate
        if (T == ComplexF16) && (n > 10)
            # is this simply too many ops for this type?
            @test_broken norm(VR*VR'-I) / (n * ulp) < vtol
            @test_broken norm(VL*VL'-I) / (n * ulp) < vtol
        else
            @test norm(VR*VR'-I) / (n * ulp) < vtol
            @test norm(VL*VL'-I) / (n * ulp) < vtol
        end
    end
end

function schurtest(A::Matrix{T}, tol; normal=false) where {T<:Complex}
    n = size(A,1)
    ulp = eps(real(T))
    # schur() uses eigtype(), which promotes ComplexF16 to ComplexF32
    if (real(T) <: BlasFloat) || (LinearAlgebra.eigtype(T) != T)
        S = GenericSchur.gschur(A)
    else
        S = schur(A)
    end
    # test 1: S.T is upper triangular
    @test all(tril(S.T,-1) .== 0)
    # test 2: norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp) < tol
    if verbosity[] > 1
        decomp_err = norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp)
        println("decomp. error: ", decomp_err)
        @test decomp_err < tol
    else
        @test norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp) < tol
    end
    # test 3: S.Z is unitary: norm(I - S.Z * S.Z') / (n * ulp) < tol
    @test norm(I - S.Z * S.Z') / (n * ulp) < tol
    # test 4: S.values are e.v. of T
    @test all(csort(S.values) .== csort(diag(S.T)))

    # It is tempting to check eigenvalues against LAPACK eigvals(A)
    # for suitable types, but the comparison is misleading without
    # detailed analysis because many of the test matrices are severly
    # non-normal.

    if normal
        # verify that Schur diagonalizes normal matrices
        @test norm(triu(S.T,1)) / (n * norm(A) * ulp) < tol
    end
    vtol = tol
    vectest(A, S, tol, normal=normal)
end

function hesstest(A::Matrix{T}, tol) where {T<:Complex}
    n = size(A,1)
    ulp = eps(real(T))
    H = invoke(GenericSchur._hessenberg!,
               Tuple{StridedMatrix{Ty},} where Ty,
               copy(A))
    Q = GenericSchur._materializeQ(H)
    if VERSION < v"1.3"
        HH = triu(H.data, -1)
    else
        HH = H.H
        # test 1: H.H is upper triangular
        @test all(tril(HH,-2) .== 0)
    end
    # test 2: norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp) < tol
    decomp_err = norm(A - Q * HH * Q') / (n * norm(A) * ulp)
    @test decomp_err < tol
    # test 3: S.Z is orthogonal: norm(I - S.Z * S.Z') / (n * ulp) < tol
    orth_err = norm(I - Q * Q') / (n * ulp)
    @test orth_err < tol
    # test 4: subdiagonal is real
    @test all(isreal.(diag(HH,-1)))
end

function hesstest(A::Hermitian{T}, tol) where {T<:Complex}
    n = size(A,1)
    ulp = eps(real(T))
    H = invoke(GenericSchur._hessenberg!,
               Tuple{Hermitian{Ty},} where {Ty <: Complex},
               copy(A))
    Q = GenericSchur._materializeQ(H)
    HH = H.H
    # test 1: H.H is formally symtridiagonal
    @test isa(HH, SymTridiagonal)
    @test eltype(HH) == real(eltype(A))
    # test 2: norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp) < tol
    decomp_err = norm(A - Q * HH * Q') / (n * norm(A) * ulp)
    @test decomp_err < tol
    # test 3: S.Z is orthogonal: norm(I - S.Z * S.Z') / (n * ulp) < tol
    orth_err = norm(I - Q * Q') / (n * ulp)
    @test orth_err < tol
end


"""
generate a random unitary matrix

Unitary matrices are normal, so Schur decomposition is diagonal for them.
(This is not only another test, but sometimes actually useful.)
"""
function randu(::Type{T},n) where {T<:Complex}
    if real(T) ∈ [Float16,Float32,Float64]
        A = randn(T,n,n)
        F = qr(A)
        return Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
    else
        # don't have normal deviates for other types, but need appropriate QR
        A = randn(ComplexF64,n,n)
        F = qr(convert.(T,A))
        return Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
    end
end

Random.seed!(1234)

@testset "Hessenberg $T" for T in [ComplexF64, Complex{Float16}]
    n = 32
    tol = 10
    A = rand(T,n,n)
    hesstest(A, tol)
    s = 100 * floatmin(real(T))
    Asmall = s * A
    hesstest(Asmall, tol)
end

if VERSION >= v"1.3"
    @testset "Hermitian Hessenberg $T" for T in [ComplexF64, Complex{Float16}]
        n = 32
        tol = 10
        A = Hermitian(rand(T,n,n), :U)
        hesstest(A, tol)
        s = 100 * floatmin(real(T))
        Asmall = s * A
        hesstest(Asmall, tol)
        A = Hermitian(rand(T,n,n), :L)
        hesstest(A, tol)
        s = 100 * floatmin(real(T))
        Asmall = s * A
        hesstest(Asmall, tol)
    end
end

@testset "group $T" for T in [Complex{BigFloat},Complex{Float16}]
unfl = floatmin(real(T))
ovfl = one(real(T)) / unfl
ulp = eps(real(T))
ulpinv = one(real(T)) / ulp
rtulp = sqrt(ulp)
rtulpi = one(real(T)) / rtulp

ens = [4,32]
@testset "random general" begin
    for n in ens
        A = rand(T,n,n)
        schurtest(A,10)
    end
end
@testset "random unitary" begin
    for n in ens
        A = randu(T,n)
        schurtest(A,20,normal=true)
    end
end

end # group testset

@testset "Godunov (complex)" begin
    # Set precision fine enough for eigval condition to apply,
    # but not fine enough to make it a slam-dunk.
    setprecision(BigFloat, 80) do
        A,v,econd = godunov(Complex{BigFloat})
        S = schur(A)
        δ = norm(S.Z*S.T*S.Z'-A)
        @test δ < 100 * eps(big(1.0)) * norm(A)
        t = 3*δ*econd
        @test isapprox(csort(S.values),v,atol=t)
        vnew = eigvals(A)
        @test isapprox(csort(vnew),v,atol=t)
    end
end

# v0.4.0 breaks on this matrix
function _example1(T=Complex{Float32})
    iseed = [4066,2905,502,2389]
    ulpinv = 1 / eps(real(T))
    A,d,ds = latme!(zeros(T,5,5),1.0,4,ulpinv,1.0,iseed=iseed)
    return A
end
@testset "short QR sweep" begin
    A = _example1(ComplexF32)
    schurtest(A,20)
end

@testset "group $T" for T in [ComplexF64, ComplexF32]

unfl = floatmin(real(T))
ovfl = one(real(T)) / unfl
ulp = eps(real(T))
ulpinv = one(real(T)) / ulp
rtulp = sqrt(ulp)
rtulpi = one(real(T)) / rtulp

# sizes of matrices to test
ens = [4,32]

itypemax = 10
# On author's dev system, tests pass with tol=5 for most and 6 for all,
# but generalization is risky.
# (LAPACK uses 20, FWIW.)
tols = fill(20,itypemax)

simrconds = [1.0, rtulpi, 0.0]
magns = [1.0, ovfl*ulp, unfl*ulpinv]

# test matrix from zdrves:
# conds=0 means index 3 for simrconds (i.e. irrelevant)
# type magn mode conds
# 1 1 0 0
# 2 1 0 0
# 3 1 0 0
# 4 1 4 0
# 4 1 3 0
# 4 1 1 0
# 4 2 4 0
# 4 3 4 0
# 6 1 4 1
# 6 1 3 1
# 6 1 1 1
# 6 1 5 1
# 6 1 4 2
# 6 1 3 2
# 6 1 1 2
# 6 1 5 2
# 6 2 5 2
# 6 3 5 2
# 9 1 4 0
# 9 2 3 0
# 9 3 1 0


# Other stuff in zdrves: pointless or not even used
# 1- zero
# 2- identity
# 4- diagonal, ev specified
# 5- symmetric, ev specified, not used in zdrves

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
#               CALL ZLATME( N, 'D', ISEED, WORK, IMODE, COND, CONE,
#     $                      'T', 'T', 'T', RWORK, 4, CONDS, N, N, ANORM,
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

@testset "triangular, specified ev" begin
    # This is done peculiarly to exercise branches in the eigenvector solver.
    # In particular, we stuff the UT directly into a Schur object so we know exactly
    # what's on the diagonal.
    # (Probably happens for normal schur(), but I didn't promise that anywhere.)
    kconds =   [2,2,2,2,2,2,2,2,2,2,2,2]
    kmagn =    [1,1,1,1,2,2,2,2,3,3,3,3]
    kmode =    [5,0,0,0,5,0,0,0,5,0,0,0]
    mydmodes = [0,0,1,2,0,0,1,2,0,0,1,2]
    nj = length(kmagn)
    itype = 6
    for n in ens
        for j in 1:nj
            A = zeros(T,n,n)
            imode = kmode[j]
            rcond = ulpinv
            anorm = magns[kmagn[j]]
            d_mode = mydmodes[j]
            (verbosity[] > 1) &&
                println("latmr: n=$n $anorm, $imode, $rcond")
            if imode == 0
                d = rand(T,n)
                if d_mode == 0
                    d[2] = d[n-1]
                elseif d_mode == 1
                    d[2] = d[n-1] * nextfloat(one(real(T)))
                else
                    d[2] = d[n-1] * prevfloat(one(real(T)))
                end
                latmr!(A, anorm, imode, rcond, kl=0, d=d)
            else
                latmr!(A, anorm, imode, rcond, kl=0)
            end
            S = LinearAlgebra.Schur(A, Diagonal(ones(T,n)), diag(A))
            vectest(A, S, tols[itype])
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
