@testset "selected errors" begin
    n = 5
    A = diagm(-1 => fill(-1.0 + 1.0im, n - 1)) + triu(rand(n, n))
    H = Hessenberg(A, zeros(n - 1))
    @test_throws ArgumentError GenericSchur.gschur!(H)

    A = triu(rand(n, n), -1)
    H = Hessenberg(A, zeros(n - 1))
    Z = rand(n - 1, n - 1)
    @test_throws DimensionMismatch GenericSchur.gschur!(H, Z)

    @test_throws DimensionMismatch GenericSchur.gschur!(rand(5, 4))
    @test_throws DimensionMismatch GenericSchur.gschur!(rand(ComplexF64, 5, 4))
end

if piracy
    # make sure our pirates aren't called for inappropriate eltypes
    @testset "letter of marque" begin
        # WARNING: These depend on absence of surprises in stdlib. Revise as needed.
        A = rand(Int, 4, 4)
        @test_throws MethodError schur!(A)
        A = rand(Int, 4, 4) + im * rand(Int, 4, 4)
        @test_throws MethodError schur!(A)
        A = rand(1:10, 4, 4) .// rand(1:10, 4, 4)
        @test_throws MethodError schur!(A)
        @test_throws MethodError hessenberg!(A)
    end
end
