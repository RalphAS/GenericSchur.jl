@testset "error handling" begin
    n = 5
    A = diagm(-1 => fill(-1.0+1.0im,n-1)) + triu(rand(n,n))
    H = GenericSchur.HessenbergArg(A,zeros(n-1))
    @test_throws ArgumentError GenericSchur.gschur!(H)

    A = triu(rand(n,n),-1)
    H = GenericSchur.HessenbergArg(A,zeros(n-1))
    Z = rand(n-1,n-1)
    @test_throws DimensionMismatch GenericSchur.gschur!(H,Z)

    @test_throws DimensionMismatch GenericSchur.gschur!(rand(5,4))
    @test_throws DimensionMismatch GenericSchur.gschur!(rand(ComplexF64,5,4))
end
