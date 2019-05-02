#module GT
#using Test, LinearAlgebra, Random
# using GenericSchur: ggschur!
#include("../src/generalized.jl")

#Random.seed!(1101)

#using Printf

function schurtest(A::Matrix{T}, B::Matrix{T}, tol;
                   normal=false, commuting=false) where {T<:Complex}
    n = size(A,1)
    ulp = eps(real(T))
#    if real(T) <: BlasFloat
        S = GenericSchur.ggschur!(copy(A), copy(B))
#    else
#        S = schur(A, B)
#    end
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


@testset "generalized sanity $T" for T in [ComplexF64]

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
        schurtest(A, B, 20)
    end
end # testset


#end # module
