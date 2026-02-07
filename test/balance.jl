function run_unbalanced(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    fu = eigen(A, scale = false)
    if T <: BlasFloat
        # use our code for balancing since that's what we're testing
        Abal, B = balance!(copy(A))
        fb1 = eigen(Abal, scale = false)
        fb = LinearAlgebra.Eigen(fb1.values, lmul!(B, fb1.vectors))
    else
        fb = eigen(A, scale = true)
    end
    if VERSION < v"1.2-"
        # compensate for sometime uncontrolled order of eigenvalues
        ewu, vu = fu
        ewb, vb = fb
        p = sortperm(real.(ewu))
        ewu = ewu[p]
        vu = vu[:, p]
        fu = LinearAlgebra.Eigen(ewu, vu)
        p = sortperm(real.(ewb))
        ewb = ewb[p]
        vb = vb[:, p]
        fb = LinearAlgebra.Eigen(ewb, vb)
    end
    return fu, fb
end

# A badly-balanced matrix based on the literature.
function unbal_classic(::Type{T}) where {T}
    m = round(-2 * log2(eps(real(T))) / 5)
    x = 2.0^m
    A = T.([1 0 x^(-2); 1 1 x^(-1); x^2 x 1])
    rt2 = sqrt(T(2))
    rth = rt2 / 2
    # warning: these are only accurate to O(1/x^3) IIUC
    λ = [1 - rt2 + 1 / (4x), 1 - 1 / (2x), 1 + rt2 + 1 / (4x)]
    v1 = [-rth / (x^2), -rth / x + 3 / (8 * x^2), 1]
    v2 = [-2 / x, 2 - 1 / (x^2), 1]
    v3 = [rth / (x^2), rth / x + 3 / (8 * x^2), 1]
    V = hcat(v1, v2, v3)
    return A, λ, V
end

if piracy
    bal_test_types = (
        Float64, ComplexF64,
        BigFloat, Complex{BigFloat},
    )
else
    bal_test_types = (
        Float64, ComplexF64,
    )
end

# Note: to avoid fragility, we only require that the worst unbalanced error be
# much larger than the worst balanced one.
# Typically all unbalanced eigenvalues are poor.
@testset "balancing $T" for T in bal_test_types
    A, λ, V = unbal_classic(T)
    fu, fb = run_unbalanced(A)
    ru = maximum([abs(fu.values[j] - λ[j]) for j in 1:3])
    rb = maximum([abs(fb.values[j] - λ[j]) for j in 1:3])
    @test ru > 100rb

    if T <: Complex
        Abal, B = balance!(copy(A))
        sb = GenericSchur.gschur(copy(Abal))
        Vl = GenericSchur.geigvecs(sb, left = true)
        ldiv!(B, Vl)
        @test A' * Vl ≈ Vl * diagm(0 => conj.(sb.values))
    end
end
@testset "balancing sanity" begin
    for n1 in [0, 1, 2]
        for n2 in [0, 1, 2]
            for n3 in [0, 1, 2]
                n = n1 + n2 + n3
                n == 0 && continue
                A1 = triu(rand(n1, n1))
                A2 = rand(n2, n2)
                A3 = triu(rand(n3, n3))
                A = [A1 zeros(n1, n2 + n3); zeros(n2, n1) A2 zeros(n2, n3); zeros(n3, n1 + n2) A3]
                Pc = zeros(n, n)
                ic = randperm(n)
                for i in 1:n
                    Pc[i, ic[i]] = 1
                end
                Ax = Pc' * A * Pc
                C, B = balance!(copy(Ax))
                @test B.ilo <= B.ihi
                nsubdiag = 0
                for i in 1:n
                    for j in 1:(i - 1)
                        if (i < B.ilo) || (j > B.ihi)
                            if C[i, j] != 0
                                nsubdiag += 1
                            end
                        end
                    end
                end
                @test nsubdiag == 0
            end # for n3
        end
    end
end
