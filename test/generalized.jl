function gschurtest(A::Matrix{T}, B::Matrix{T}, tol;
                   normal=false, commuting=false) where {T<:Complex}
    n = size(A,1)
    ulp = eps(real(T))
    if real(T) <: BlasFloat
        S = GenericSchur.ggschur!(copy(A), copy(B))
    else
        S = schur(A, B)
    end
    # test 1: S.T is upper triangular
    @test norm(tril(S.S,-1)) / (n * norm(A) * ulp) < tol
    @test norm(tril(S.T,-1)) / (n * norm(A) * ulp) < tol
    # @test all(tril(S.S,-1) .== 0)
    # @test all(tril(S.T,-1) .== 0)
    # test 2: norm(A - S.Q * S.T * S.Z') / (n * norm(A) * ulp) < tol
    @test norm(A - S.Q * S.S * S.Z') / (n * norm(A) * ulp) < tol
    @test norm(B - S.Q * S.T * S.Z') / (n * norm(A) * ulp) < tol
    # test 3: S.Z is unitary: norm(I - S.Z * S.Z') / (n * ulp) < tol
    @test norm(I - S.Z * S.Z') / (n * ulp) < tol
    @test norm(I - S.Q * S.Q') / (n * ulp) < tol
    # test 4: S.values are e.v. of T
#    @test all(csort(S.values) .== csort(diag(S.T)))

    # It is tempting to check eigenvalues against LAPACK eigvals(A)
    # for suitable types, but the comparison is misleading without
    # detailed analysis because many of the test matrices are severly
    # non-normal.

    if normal && commuting
        # verify that Schur diagonalizes normal commuting matrices
        @test norm(triu(S.S,1)) / (n * norm(B) * ulp) < tol
        @test norm(triu(S.T,1)) / (n * norm(A) * ulp) < tol
    end

    VR = eigvecs(S)
    evcheck = norm((A * VR) * diagm(0 => S.β) - (B * VR) * diagm(0 => S.α))
    @test evcheck / (n*norm(A) * ulp) < tol
    # someday maybe left eigenvectors too
end


@testset "generalized basic sanity $T" for T in [ComplexF64, Complex{BigFloat}]

unfl = floatmin(real(T))
ovfl = one(real(T)) / unfl
ulp = eps(real(T))
ulpinv = one(real(T)) / ulp
rtulp = sqrt(ulp)
rtulpi = one(real(T)) / rtulp

# sizes of matrices to test
ens = [4,32]

    for n in ens
        A = rand(T,n,n)
        B = rand(T,n,n)
        gschurtest(A, B, 20)
    end
end # testset

using SparseArrays
include("psjmats.jl")
@testset "generalized examples from Q.M." begin
    gschurtest(Array(psj_1_A), Array(psj_1_B), 100)
    gschurtest(Array(psj_2_A), Array(psj_2_B), 100)
    gschurtest(Array(psj_3_A), Array(psj_3_B), 100)
end

@testset "generalized challenging $T" for T in [ComplexF64, Complex{BigFloat}]
    # sizes of matrices to test
    n = 32
    n1 = n >> 2
    n2 = n - 2*n1
    A0 = triu(rand(T,n,n) .- 0.5)
    B0 = triu(rand(T,n,n) .- 0.5)
    # stuff in some degenerate singularity
    for j=1:n1
        A0[j,j] = zero(T)
    end
    for j=n1+1:n2
        B0[j,j] = zero(T)
    end
    Qa,_ = qr!(T.(randn(n,n) + 1im * randn(n,n)))
    Qb,_ = qr!(T.(randn(n,n) + 1im * randn(n,n)))
    A = Qa*A0*Qa'
    B = Qb*B0*Qb'

    gschurtest(A, B, 20)

end # testset


#end # module
