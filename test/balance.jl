function run_unbalanced(A::AbstractMatrix{T}) where T
    n = size(A,1)
    fu = eigen(A, scale=false)
    if T <: BlasFloat
        # use our code for balancing since that's what we're testing
        Abal,  B = balance!(copy(A))
        fb1 = eigen(Abal, scale=false)
        fb = LinearAlgebra.Eigen(fb1.values, lmul!(B, fb1.vectors))
    else
        fb = eigen(A, scale=true)
    end
    if VERSION < v"1.2-"
        # compensate for sometime uncontrolled order of eigenvalues
        ewu, vu = fu
        ewb, vb = fb
        p = sortperm(real.(ewu))
        ewu = ewu[p]
        vu = vu[:,p]
        fu = LinearAlgebra.Eigen(ewu,vu)
        p = sortperm(real.(ewb))
        ewb = ewb[p]
        vb = vb[:,p]
        fb = LinearAlgebra.Eigen(ewb,vb)
    end
    fu, fb
end

# A badly-balanced matrix based on the literature.
function unbal_classic(::Type{T}) where T
    m = round(-2*log2(eps(real(T)))/5)
    x = 2.0^m
    A = T.([1 0 x^(-2); 1 1 x^(-1); x^2 x 1])
    rt2 = sqrt(T(2))
    rth = rt2/2
    # warning: these are only accurate to O(1/x^3) IIUC
    λ = [1-rt2+1/(4x),1-1/(2x),1+rt2+1/(4x)]
    v1 = [-rth/(x^2), -rth/x+3/(8*x^2), 1]
    v2 = [-2/x, 2-1/(x^2), 1]
    v3 = [rth/(x^2), rth/x+3/(8*x^2), 1]
    V = hcat(v1,v2,v3)
    A, λ, V
end

# Note: to avoid fragility, we only require that the worst unbalanced error be
# much larger than the worst balanced one.
# Typically all unbalanced eigenvalues are poor.
@testset "balancing $T" for T in (Float64, ComplexF64,
                                  BigFloat, Complex{BigFloat})
    A, λ, V = unbal_classic(T)
    fu, fb = run_unbalanced(A)
    ru = maximum([abs(fu.values[j]-λ[j]) for j in 1:3])
    rb = maximum([abs(fb.values[j]-λ[j]) for j in 1:3])
    @test ru > 100rb
end

# This is a convenient place for other checks of the simple wrapper.
# We shouldn't test StdLib code here, so no BlasFloat.
@testset "vector normalization $T" for T in (BigFloat, Complex{BigFloat})
    m = round(-log10(eps(real(T)))/3)
    A = T.([1 0 10.0^(-2m); 1 1 10.0^(-m); 10.0^(2m) 10.0^m 1])
    E,V = eigen(A, scale=true)
    tol = 10
    u = eps(real(T))
    n = size(A,1)
    for j in 1:n
        v = V[:,j]
        @test abs(norm(v,2) - 1) / (n*u) < tol
    end
end
