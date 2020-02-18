module GLAPACK
# missing or inadequate wrappers: tgsen, tgsyl
using LinearAlgebra

const liblapack = Base.liblapack_name

import LinearAlgebra.BLAS.@blasfunc

import LinearAlgebra: BlasFloat, BlasInt, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, checksquare

import LinearAlgebra.LAPACK: chkargsok, chklapackerror, chktrans

for (tgsen, elty) in
    ((:dtgsen_, :Float64),
     (:stgsen_, :Float32))
    @eval begin
        function tgsenx!(select::AbstractVector{BlasInt},
                         S::AbstractMatrix{$elty}, T::AbstractMatrix{$elty},
                         Q::AbstractMatrix{$elty}, Z::AbstractMatrix{$elty};
                         pnorm = :Frobenius)
            chkstride1(select, S, T, Q, Z)
            n, nt, nq, nz = checksquare(S, T, Q, Z)
            if n != nt
                throw(DimensionMismatch("dimensions of S, ($n,$n), and T, ($nt,$nt), must match"))
            end
            if n != nq
                throw(DimensionMismatch("dimensions of S, ($n,$n), and Q, ($nq,$nq), must match"))
            end
            if n != nz
                throw(DimensionMismatch("dimensions of S, ($n,$n), and Z, ($nz,$nz), must match"))
            end
            lds = max(1, stride(S, 2))
            ldt = max(1, stride(T, 2))
            ldq = max(1, stride(Q, 2))
            ldz = max(1, stride(Z, 2))
            m = Ref{BlasInt}(sum(select))
            alphai = similar(T, $elty, n)
            alphar = similar(T, $elty, n)
            beta = similar(T, $elty, n)
            lwork = BlasInt(-1)
            work = Vector{$elty}(undef, 1)
            liwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 1)
            pl = Ref{$elty}()
            pr = Ref{$elty}()
            dif = Vector{$elty}(undef, 2)
            info = Ref{BlasInt}()
            select = convert(Array{BlasInt}, select)
            ijob = BlasInt((pnorm == :Frobenius) ? 4 : 5)
            for i = 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                ccall((@blasfunc($tgsen), liblapack), Cvoid,
                       (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                        Ptr{BlasInt}),
                    ijob, 1, 1, select,
                    n, S, lds, T,
                    ldt, alphar, alphai, beta,
                    Q, ldq, Z, ldz,
                    m, pl, pr, dif,
                    work, lwork, iwork, liwork,
                    info)
                chklapackerror(info[])
                if i == 1 # only estimated optimal lwork, liwork
                    lwork  = 2*BlasInt(real(work[1]))
                    resize!(work, lwork)
                    liwork = BlasInt(real(iwork[1]))
                    resize!(iwork, liwork)
                end
            end
            S, T, complex.(alphar, alphai), beta, Q, Z, pl[], pr[], dif
        end
    end
end

for (tgsen, elty, relty) in
    ((:ztgsen_, :ComplexF64, :Float64),
     (:ctgsen_, :ComplexF32, :Float32))
    @eval begin
        function tgsenx!(select::AbstractVector{BlasInt},
                         S::AbstractMatrix{$elty}, T::AbstractMatrix{$elty},
                         Q::AbstractMatrix{$elty}, Z::AbstractMatrix{$elty};
                         pnorm = :Frobenius)
            chkstride1(select, S, T, Q, Z)
            n, nt, nq, nz = checksquare(S, T, Q, Z)
            if n != nt
                throw(DimensionMismatch("dimensions of S, ($n,$n), and T, ($nt,$nt), must match"))
            end
            if n != nq
                throw(DimensionMismatch("dimensions of S, ($n,$n), and Q, ($nq,$nq), must match"))
            end
            if n != nz
                throw(DimensionMismatch("dimensions of S, ($n,$n), and Z, ($nz,$nz), must match"))
            end
            lds = max(1, stride(S, 2))
            ldt = max(1, stride(T, 2))
            ldq = max(1, stride(Q, 2))
            ldz = max(1, stride(Z, 2))
            m = Ref{BlasInt}(sum(select))
            alpha = similar(T, $elty, n)
            beta = similar(T, $elty, n)
            lwork = BlasInt(-1)
            work = Vector{$elty}(undef, 1)
            liwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 1)
            pl = Ref{$relty}()
            pr = Ref{$relty}()
            dif = Vector{$relty}(undef, 2)
            info = Ref{BlasInt}()
            select = convert(Array{BlasInt}, select)
            ijob = BlasInt((pnorm == :Frobenius) ? 4 : 5)
            for i = 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                ccall((@blasfunc($tgsen), liblapack), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ptr{$relty}, Ptr{$relty}, Ptr{$relty},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                       Ptr{BlasInt}),
                      ijob, 1, 1, select,
                      n, S, lds, T,
                      ldt, alpha, beta,
                      Q, ldq, Z, ldz,
                      m, pl, pr, dif,
                      work, lwork, iwork, liwork,
                      info)
                chklapackerror(info[])
                if i == 1 # only estimated optimal lwork, liwork
                    lwork  = 2*BlasInt(real(work[1]))
                    resize!(work, lwork)
                    liwork = BlasInt(real(iwork[1]))
                    resize!(iwork, liwork)
                end
            end
            if m[] != sum(select)
                @warn "surprise! tgsen returns m=$(m[]), expected $(sum(select))"
            end
            S, T, alpha, beta, Q, Z, pl[], pr[], dif
        end # func
    end # eval
end # type loop

for (fn, elty, relty) in ((:dtgsyl_, :Float64, :Float64),
                   (:stgsyl_, :Float32, :Float32),
                   (:ztgsyl_, :ComplexF64, :Float64),
                   (:ctgsyl_, :ComplexF32, :Float32))
    @eval begin
        function tgsyl!(transa::AbstractChar,
                        A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty},
                        C::AbstractMatrix{$elty}, D::AbstractMatrix{$elty},
                        E::AbstractMatrix{$elty}, F::AbstractMatrix{$elty},
                        ijob::Int=1)
            # require_one_based_indexing(A, B, C)
            chkstride1(A, B, C)
            m, n = checksquare(A, B)
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            m1, n1 = size(C)
            if m != m1 || n != n1
                throw(DimensionMismatch("dimensions of A, ($m,$n), and C, ($m1,$n1), must match"))
            end
            ldc = max(1, stride(C, 2))

            # require_one_based_indexing(D, E, F)
            chkstride1(D, E, F)
            m2, n2 = checksquare(D, E)
            if m != m2 || n != n2
                throw(DimensionMismatch("dimensions of (A,B), ($m,$n), and (D,E), ($m2,$n2), must match"))
            end
            ldd = max(1, stride(D, 2))
            lde = max(1, stride(E, 2))
            m1, n1 = size(F)
            if m != m1 || n != n1
                throw(DimensionMismatch("dimensions of A, ($m,$n), and F, ($m1,$n1), must match"))
            end
            ldf = max(1, stride(F, 2))

            scale = Ref{$relty}(1.0)
            dif = Ref{$relty}(0.0)
            info  = Ref{BlasInt}()
            lwork = BlasInt(-1)
            work = Vector{$elty}(undef, 1)
            # only need m+n+2 for complex but m+n+6 for real
            iwork = Vector{BlasInt}(undef, m+n+6)
            for i=1:2 # first call to get workspace
              ccall((@blasfunc($fn), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{BlasInt}, # trans, ijob
                   Ref{BlasInt}, Ref{BlasInt}, # m,n
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$relty}, Ptr{$relty}, # scale, dif
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, # work, lwork, iwork
                   Ptr{BlasInt}),
                    transa, ijob, m, n,
                    A, lda, B, ldb, C, ldc,
                    D, ldd, E, lde, F, ldf,
                    scale, dif, work, lwork, iwork, info)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            C, F, scale[], dif[]
        end # func
    end # eval
end # type loop

end # module
