using GenericSchur: trsylvester!, adjtrsylvester!
using GenericSVD

Random.seed!(1234)

function checkgsyl(m=4,n=2,Ty=Float64)
    rctr = Ty(0.5) * ((Ty <: Complex) ? (1+1im) : 1)
    A = triu(rand(Ty,m,m) .- rctr) + 2I
    B = triu(rand(Ty,n,n) .- rctr) + I
    X = rand(Ty,m,n)

    # separate spectra so problem is not too hard
    D = triu(rand(Ty,m,m) .- rctr) + 4I
    E = triu(rand(Ty,n,n) .- rctr) + I
    Y = rand(Ty,m,n)
    C = A*X-Y*B
    C2 = copy(C)
    F = D*X-Y*E
    F2 = copy(F)

    X2,Y2,scale = trsylvester!(A,B,C2,D,E,F2)
    fwd2a = norm(scale*X-X2) / norm(X)
    fwd2b = norm(scale*Y-Y2) / norm(Y)
    bwd2a = norm(A*X2-Y2*B-scale*C)/((norm(A)+norm(B))*norm(X))
    bwd2b = norm(D*X2-Y2*E-scale*F)/((norm(D)+norm(E))*norm(X))
    if verbosity[] > 0
        println("our    scale $scale error norms fwd: ",fwd2a," ",fwd2b)
        println("                                bwd: ", bwd2a," ",bwd2b)
    end
    tol = 2
    stol = tol * max(m,n) * eps(real(Ty))
    # forward errors for Sylvester are fiddly
    if max(m,n) < 10
        @test fwd2a < sqrt(stol)
        @test fwd2b < sqrt(stol)
    end
    @test bwd2a < tol
    @test bwd2b < tol

end

function checkgasyl(m=4,n=2,Ty=Float64)
    A = triu(rand(Ty,m,m) .- 0.5) + 2I
    B = triu(rand(Ty,n,n) .- 0.5) + I
    X = rand(Ty,m,n)

    # separate spectra so problem is not too hard
    D = triu(rand(Ty,m,m) .- 0.5) + 4I
    E = triu(rand(Ty,n,n) .- 0.5) + I
    Y = rand(Ty,m,n)
    C = A'*X + D'*Y
    C2 = copy(C)
    F = -X*B' - Y*E'
    F2 = copy(F)

    X2, Y2, scale = adjtrsylvester!(A,B,C2,D,E,F2)
    fwd2a = norm(scale*X-X2) / norm(X)
    fwd2b = norm(scale*Y-Y2) / norm(Y)
    bwd2a = norm(A'*X2+D'*Y2-scale*C)/((norm(A)+norm(B))*norm(X))
    bwd2b = norm(X2*B'+Y2*E'+scale*F)/((norm(D)+norm(E))*norm(X))
    if verbosity[] > 0
        println("our    scale $scale error norms fwd: ",fwd2a," ",fwd2b)
        println("                                bwd: ", bwd2a," ",bwd2b)
    end

    tol = 2
    stol = tol * max(m,n) * eps(real(Ty))
    if max(m,n) < 10
        @test fwd2a < sqrt(stol)
        @test fwd2b < sqrt(stol)
    end
    @test bwd2a < tol
    @test bwd2b < tol

end

for Ty in [ComplexF64, Float64, Complex{BigFloat}]
    @testset "gen. sylvester $Ty" begin
        checkgsyl(2,4,Ty)
        checkgsyl(4,2,Ty)
        checkgsyl(24,24,Ty)
    end
    @testset "gen. adj. sylvester $Ty" begin
        checkgasyl(2,4,Ty)
        checkgasyl(4,2,Ty)
        checkgasyl(24,24,Ty)
    end
end

function checkord(A::Matrix{Ty}, B::Matrix{Ty}, tol=10) where {Ty<:Complex}
    n = size(A,1)
    ulp = eps(real(Ty))

    GS = schur(A,B)
    S2 = copy(GS.S)
    Q2 = copy(GS.Q)
    T2 = copy(GS.T)
    Z2 = copy(GS.Z)
    select = fill(true,n)
    for i=1:(n>>1)
        select[i] = false
    end
    # we extend the LinearAlgebra function, but want to test
    # our version with BLAS types:
    S2,T2,a,b,Q2,Z2 = invoke(LinearAlgebra._ordschur!,
                             Tuple{StridedMatrix{T}, StridedMatrix{T},
                                   StridedMatrix{T}, StridedMatrix{T},
                                   Union{Vector{Bool},BitVector}}
                             where T <: Complex{Tr} where Tr <: AbstractFloat,
                           S2,T2,Q2,Z2,select)

    # usual tests for Schur
    @test norm(tril(S2,-1)) / (n * norm(A) * ulp) < tol
    @test norm(tril(T2,-1)) / (n * norm(A) * ulp) < tol
    # @test all(tril(S.S,-1) .== 0)
    # @test all(tril(S.T,-1) .== 0)
    # test 2: norm(A - S.Q * S.T * S.Z') / (n * norm(A) * ulp) < tol
    @test norm(A - Q2 * S2 * Z2') / (n * norm(A) * ulp) < tol
    @test norm(B - Q2 * T2 * Z2') / (n * norm(B) * ulp) < tol
    # test 3: S.Z is unitary: norm(I - S.Z * S.Z') / (n * ulp) < tol
    @test norm(I - Z2 * Z2') / (n * ulp) < tol
    @test norm(I - Q2 * Q2') / (n * ulp) < tol

    # make sure we got the ones we asked for
    # WARNING: this test may fail if selected ev are ill-conditioned
    nwanted = count(select)
    wwanted = [GS.values[j] for j=1:n if select[j]]
    errs = zeros(nwanted)
    for i=1:nwanted
        w = a[i] / b[i]
        t,j = findmin(abs.(wwanted .- w))
        errs[i] =  t / (ulp + abs(w))
        wwanted = deleteat!(wwanted, j)
    end
    @test all(errs .< tol)
end

for Ty in [ComplexF64, Complex{BigFloat}]
    @testset "gen. ordschur $Ty" begin
        n = 16
        A = rand(Ty,n,n)
        rctr = Ty(0.5) * ((Ty <: Complex) ? (1+1im) : 1)
        B = rand(Ty,n,n) .- rctr + I
        checkord(A,B)
    end
end

# WARNING: KPTest was adapted from a more thorough validation study.
# For interesting arguments the tests don't pass as written, but results
# agree with LAPACK. With benign arguments, however, it is a good
# test of functionality.

# Test with CHK4 matrices from Kågström and Poromaa, LAWN 87, 1994.
function KPTest(α,β,x,y,dtype,Ty=ComplexF64)
    Db = I
    if dtype == 1
        Da = diagm(0=>Ty.([1,2,3,4,5])) + α * I
        atrue = [1,2,3,4,5] .+ α
    else
        Da = diagm(0=>Ty.([1,1,1,1+α,1+α]), 1=>Ty.([-1,0,0,1+β]),
                   -1=>Ty.([1,0,0,-1-β]))
        atrue = [1-1im,1+1im,1,1+α-(1+β)*im,1+α+(1+β)*im]
    end
    YH = diagm(0=>ones(Ty,5), 1=>Ty.([0,-y,0,0]), 2=>Ty.([-y,y,0]),
               3=>Ty.([y,-y]), 4=>Ty.([-y]))
    X = diagm(0=>ones(Ty,5), 1=>Ty.([0,x,0,0]), 2=>Ty.([-x,-x,0]),
              3=>Ty.([-x,-x]), 4=>Ty.([x]))
    A = YH \ (Da / X)
    B = YH \ (Db / X)
    S  = schur(A,B)
    aest = S.α ./ S.β
    for j=1:5
        λtrue = atrue[j]
        _,idx = findmin(abs.(aest .- λtrue))
        sel = falses(5)
        sel[idx] = true
        if idx == 1
            S2 = S
        else
            S2 = ordschur(S,sel)
        end
        pl, pr = eigvalscond(S2, 1)
        sep =  subspacesep(S2, 1)

        # known values from construction
        if dtype == 1 || j==3
            y = YH[j,:] # a column vector!
            x = X[:,j]
        else
            if j==1
                y = YH[1,:] + im * YH[2,:]
                x = X[:,1] + im * X[:,2]
            elseif j==2
                y = YH[1,:] - im * YH[2,:]
                x = X[:,1] - im * X[:,2]
            elseif j==4
                y = YH[4,:] - im * YH[5,:]
                x = X[:,4] - im * X[:,5]
            elseif j==5
                y = YH[4,:] + im * YH[5,:]
                x = X[:,4] + im * X[:,5]
            end
        end

        rcond = (sqrt(abs(dot(y,(A * x)))^2 + abs(dot(y,(B * x)))^2)
                 / (norm(x)*norm(y'))) / sqrt(3.0)

        n = 5
        Inm1 = Matrix{Ty}(I,n-1,n-1)
        Zl = hcat(vcat(λtrue * Inm1, Inm1),
                  vcat(-S2.S[2:n,2:n], -S2.T[2:n,2:n]))
        difl = minimum(svdvals(Zl))
        if verbosity[] > 0
            println("λ true $λtrue error $(λtrue - aest[idx])")
            println("ours   pl $(pl) pr $(pr) sep $(sep)")
            println("true rcond $rcond fudged $(rcond)  Difl $(difl)")
        end

        @test abs(log10(pl / rcond)) < 2
        @test abs(log10(pr / rcond)) < 2
        @test abs(log10(sep[1]/difl)) < 2
        @test abs(log10(sep[2]/difl)) < 2

    end
end

for Ty in [ComplexF64, Complex{BigFloat}]
    @testset "gen. condition $Ty" begin
        α = 1.0
        β = 1.0
        x = 1.0
        y = 1.0
        dtype = 2
        KPTest(α,β,x,y,dtype,Ty)
    end
end
