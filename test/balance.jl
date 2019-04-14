function run_unbalanced(A::AbstractMatrix{T}) where T
    n = size(A,1)
    fu = eigen(A, scale=false)
    if T <: BlasFloat
        # use our code for balancing
        Abal,  B = balance!(copy(A))
        fb1 = eigen(Abal)
        fb = LinearAlgebra.Eigen(fb1.values, lmul!(B, fb1.vectors))
    else
        fb = eigen(A, scale=true)
    end
    resids(E) = [norm(A*E.vectors[:,j]-E.values[j]*E.vectors[:,j])/abs(E.values[j]) for j in 1:n]
    ru = maximum(resids(fu))
    rb = maximum(resids(fb))
    # println("residuals: bare $ru balanced $rb")
    ru, rb
end
@testset "balancing $T" for T in (Float64, ComplexF64,
                                  BigFloat, Complex{BigFloat})
    m = round(-log10(eps(real(T)))/3)
    A = T.([1 0 10.0^(-2m); 1 1 10.0^(-m); 10.0^(2m) 10.0^m 1])
    ru, rb = run_unbalanced(A)
    @test ru > rb
end
