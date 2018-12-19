const badevs = Ref{Int}(0)
using Printf
function schurtest(A::Matrix{T}, tol; normal=false) where {T<:Complex}
    n = size(A,1)
    ulp = eps(real(T))
    if real(T) <: BlasFloat
        S = GenericSchur.gschur(A)
    else
        S = schur(A)
    end
    # test 1: S.T is upper triangular
    @test all(tril(S.T,-1) .== 0)
    # test 2: norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp) < tol
    @test norm(A - S.Z * S.T * S.Z') / (n * norm(A) * ulp) < tol
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
    # CHECKME: a few hard cases fail; is overflow protection sufficient?
    fom = norm(A) / (floatmin(real(T))/eps(real(T)))
    FOM = norm(A) / (floatmax(real(T))*eps(real(T)))
    if (fom > 25) && (FOM < 0.04)
        As = A
        Ss = S
        vtol = tol
    else
        vtol = (T == Complex{Float16} ? 200 : 20)
        # test work-arounds
        if fom <= 25
            As = A * (floatmax(real(T))*eps(real(T)))
            Ss = GenericSchur.gschur(As)
        elseif FOM >= 0.04
            As =  A * (floatmin(real(T))/eps(real(T)))
            Ss = GenericSchur.gschur(As)
        end
    end
    VR = eigvecs(Ss)
    @test norm(As * VR - VR * diagm(0 => Ss.values)) / (n * norm(As) * ulp) < vtol
    VL = eigvecs(Ss, left=true)
    @test norm(As' * VL - VL * diagm(0 => conj.(Ss.values))) / (n * norm(As) * ulp) < vtol
end


"""
generate a random unitary matrix

Just something good enough for present purposes.
Unitary matrices are normal, so Schur decomposition is diagonal for them.
(This is not only another test, but sometimes actually useful.)
"""
function randu(::Type{T},n) where {T<:Complex}
    if T ∈ [Float16,Float32,Float64]
        A = randn(T,n,n)
        F = qr(A)
        return Matrix(F.Q)
    else
        # don't have normal deviates for other types, but need appropriate QR
        A = randn(ComplexF64,n,n)
        F = qr(convert.(T,A))
        return Matrix(F.Q)
    end
end

Random.seed!(1234)

for T in [Complex{BigFloat},Complex{Float16}]
@testset "group $T" begin
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
        schurtest(A,10,normal=true)
    end
end

end # group testset
end # type loop (non-BlasFloat)

@testset "Godunov (complex)" begin
    setprecision(BigFloat, 80) do
        A,v = godunov(Complex{BigFloat})
        S = schur(A)
        @test isapprox(csort(S.values),v,atol=2e-5)
        vnew = eigvals(A)
        @test isapprox(csort(vnew),v,atol=2e-5)
    end
end

for T in [ComplexF64, ComplexF32]
@testset "group $T" begin

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
