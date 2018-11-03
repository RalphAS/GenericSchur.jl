module TMGlib
# wrap some routines from the LAPACK test matrix generator library
export latme!, latmr!

using LinearAlgebra
import LinearAlgebra.BlasInt
import LinearAlgebra.BLAS.@blasfunc

# CHECKME: routines linked here are included in many BLAS/LAPACK distributions
# (OpenBLAS, MKL), but do we need to allow for others?

const liblapack = Base.liblapack_name

const latm_modes = Dict(:specified => 0, :clustered_small => 1,
                    :clustered_large => 2, :exponential => 3,
                    :arithmetic => 4, :random_log => 5, :random => 6)

"""
construct a random non-symmetric square matrix with specified eigenvalues

`latme!(A::Matrix{Complex{T}}, anorm::Real, imode::Int, rcond::Real,
        simrcond::Real = 1.0;
        iseed::Vector{BlasInt} = [2518, 3899, 995, 397],
        kl::Int = size(A,2), ku::Int = size(A,2),
        dist::AbstractChar = 'D', rsign::Bool = true,
        upper::Bool = true, simtrans::Bool = true,
        simmode::Int = 4)`

`imode`: 1 - clustered small, 2 - clustered large, 3 - exponential,
       4 - arithmetic, 5 - random log, 6 - random (<0 for reversed order)
`imode = 0`, explicitly specified eigenvalues, is not yet implemented.
`rcond`: ratio of extreme eigenvalues
`kl,ku`: lower and upper bandwidths

If `simtrans == 'T'` apply a similarity transform with singular values
specified by `simmode, simrcond`, similarly to `imode,rcond`.

`dist`: character key for distribution to be sampled: U - uniform(0,1),
   S - symmetric uniform(-1,1), N - normal, D - uniform on complex unit disk.
"""
function latme! end

for (fname, elty, relty) in ((:zlatme_, :ComplexF64, :Float64),
                             (:clatme_, :ComplexF32, :Float32))
    @eval begin
        function latme!(A::Matrix{$elty}, anorm::Real, imode::Int, rcond::Real,
                simrcond::Real = 1.0;
                iseed::Vector{BlasInt} = BlasInt[2518, 3899, 995, 397],
                kl::Int = size(A,2), ku::Int = size(A,2),
                dist::AbstractChar = 'D', rsign::Bool = true,
                upper::Bool = true, simtrans::Bool = true,
                simmode::Int = 4,
                dmax::$elty = one($elty),
                d::Vector{$elty} = Vector{$elty}(undef,size(A,2)),
                ds::Vector{$relty} = Vector{$relty}(undef,size(A,2))
                )
            n = size(A,2)
            lda = size(A,1)

            if imode == 0
                (length(d) == n) || throw(ArgumentError("imode = 0 needs explicit d of length n"))
                if simtrans
                    (length(ds) == n) || throw(ArgumentError("imode = 0 needs explicit ds of length n"))
                end
            end
            (dist ∈ ['D','U','S','N']) || throw(ArgumentError("allowed values of dist are 'D','U','S', and 'N'"))

            work = Vector{$elty}(undef, 3*n)
            info = Ref{BlasInt}(0)

            ccall((@blasfunc($fname), liblapack), Cvoid,
                  (Ref{BlasInt}, # n
                   Ref{UInt8}, # dist
                   Ptr{BlasInt}, # iseed
                   Ptr{$elty}, # d(w)
                   Ref{BlasInt}, # mode
                   Ref{$relty}, # cond
                   Ref{$elty}, # dmax
                   Ref{UInt8}, # rsign
                   Ref{UInt8}, # upper
                   Ref{UInt8}, # sim
                   Ptr{$relty}, # ds(w)
                   Ref{BlasInt}, # modes
                   Ref{$relty}, # conds
                   Ref{BlasInt}, # kl
                   Ref{BlasInt}, # ku
                   Ref{$relty}, # anorm
                   Ptr{$elty}, # A
                   Ref{BlasInt}, # lda
                   Ptr{$elty}, # work(w)
                   Ptr{BlasInt} # info(w)
                   ),
                  n, dist, iseed, d, imode, rcond, dmax,
                  rsign ? 'T' : 'F', upper ? 'T' : 'F', simtrans ? 'T' : 'F',
                  ds, simmode, simrcond, kl, ku, anorm, A, lda, work, info)
            (info[] == 0) || error("info = $(info[]) from latme!")
            A, d, ds, iseed
        end
    end
end

# Note LAPACK routine has different arg list for the real versions:
for (fname, celty, elty) in ((:dlatme_, :ComplexF64, :Float64),
                             (:slatme_, :ComplexF32, :Float32))
    @eval begin
        function latme!(A::Matrix{$elty}, anorm::Real, imode::Int, rcond::Real,
                simrcond::Real = 1.0;
                iseed::Vector{BlasInt} = BlasInt[2518, 3899, 995, 397],
                kl::Int = size(A,2), ku::Int = size(A,2),
                dist::AbstractChar = 'S', rsign::Bool = true,
                upper::Bool = true, simtrans::Bool = true,
                simmode::Int = 4,
                dmax::$elty = one($elty),
                d::Vector{$elty} = Vector{$elty}(undef,size(A,2)),
                ds::Vector{$elty} = Vector{$elty}(undef,size(A,2))
                )
            n = size(A,2)
            lda = size(A,1)

            if imode == 0
                (length(d) == n) || throw(ArgumentError("imode = 0 needs explicit d of length n"))
                if simtrans
                    (length(ds) == n) || throw(ArgumentError("imode = 0 needs explicit ds of length n"))
                end
            end
            (dist ∈ ['U','S','N']) || throw(ArgumentError("allowed values of dist are 'U','S', and 'N'"))

            work = Vector{$elty}(undef, 3*n)
            info = Ref{BlasInt}(0)

            # if specifying eigenvalues
#           EI is CHARACTER*1 array, dimension ( N )
#            If MODE is 0, and EI(1) is not ' ' (space character),
#            this array specifies which elements of D (on input) are
#            real eigenvalues and which are the real and imaginary parts
#            of a complex conjugate pair of eigenvalues.  The elements
#            of EI may then only have the values 'R' and 'I'.  If
#            EI(j)='R' and EI(j+1)='I', then the j-th eigenvalue is
#            CMPLX( D(j) , D(j+1) ), and the (j+1)-th is the complex
#            conjugate thereof.  If EI(j)=EI(j+1)='R', then the j-th
#            eigenvalue is D(j) (i.e., real).  EI(1) may not be 'I',
#            nor may two adjacent elements of EI both have the value 'I'.
#            If MODE is not 0, then EI is ignored.  If MODE is 0 and
#            EI(1)=' ', then the eigenvalues will all be real.

            ei = ' '

            ccall((@blasfunc($fname), liblapack), Cvoid,
                  (Ref{BlasInt}, # n
                   Ref{UInt8}, # dist
                   Ptr{BlasInt}, # iseed
                   Ptr{$elty}, # d(w)
                   Ref{BlasInt}, # mode
                   Ref{$elty}, # cond
                   Ref{$elty}, # dmax
                   Ref{UInt8}, # ei
                   Ref{UInt8}, # rsign
                   Ref{UInt8}, # upper
                   Ref{UInt8}, # sim
                   Ptr{$elty}, # ds(w)
                   Ref{BlasInt}, # modes
                   Ref{$elty}, # conds
                   Ref{BlasInt}, # kl
                   Ref{BlasInt}, # ku
                   Ref{$elty}, # anorm
                   Ptr{$elty}, # A
                   Ref{BlasInt}, # lda
                   Ptr{$elty}, # work(w)
                   Ptr{BlasInt} # info(w)
                   ),
                  n, dist, iseed, d, imode, rcond, dmax, ei,
                  rsign ? 'T' : 'F', upper ? 'T' : 'F', simtrans ? 'T' : 'F',
                  ds, simmode, simrcond, kl, ku, anorm, A, lda, work, info)
            (info[] == 0) || error("info = $(info[]) from latme!")
            A, d, ds, iseed
        end
    end
end

"""
construct a random matrix of various types

latmr!(A::Matrix{T<:BLAStype}, anorm::Real, imode::Int, rcond::Real,
                rcondr::Real = 1.0, rcondl::Real = 1.0;
                sym::AbstractChar = 'N',
                iseed::Vector{BlasInt} = [2518, 3899, 995, 397],
                kl::Int = size(A,2), ku::Int = size(A,2),
                dist::AbstractChar = 'D', rsign::Bool = true,
                grade::AbstractChar = 'N', model::Int = 4, moder::Int = 4,
                pivtng::AbstractChar = 'N', ipivot::Vector{BlasInt} = BlasInt[],
                sparse::Real = 0.0,
                pack::AbstractChar = 'N'
                )

`imode`: 1 - clustered small, 2 - clustered large, 3 - exponential,
       4 - arithmetic, 5 - random log, 6 - random (<0 for reversed order)
`imode = 0`, explicitly specified eigenvalues, is not yet implemented.
`sym`: `'S'` - symmetric, `'H'` - Hermitian, `'N'` - nonsymmetric
`rcond`: ratio of extreme eigenvalues
`kl,ku`: lower and upper bandwidths
`sparse`: fraction to be nulled out
`dist`: character key for distribution to be sampled: U - uniform(0,1),
   S - symmetric uniform(-1,1), N - normal, D - uniform on complex unit disk.
"""
function latmr! end

for (fname, elty, relty) in ((:zlatmr_, :ComplexF64, :Float64),
                             (:clatmr_, :ComplexF32, :Float32),
                             (:dlatmr_, :Float64, :Float64),
                             (:slatmr_, :Float32, :Float32))
    @eval begin
        function latmr!(A::Matrix{$elty}, anorm::Real, imode::Int, rcond::Real,
                        rcondr::Real = 1.0, rcondl::Real = 1.0;
                        sym::AbstractChar = 'N',
                        iseed::Vector{BlasInt} = BlasInt[2518, 3899, 995, 397],
                        kl::Int = size(A,2), ku::Int = size(A,2),
                        dist::AbstractChar = $elty <: Complex ? 'D' : 'S',
                        rsign::Bool = true,
                        grade::AbstractChar = 'N', model::Int = 1, moder::Int = 1,
                        pivtng::AbstractChar = 'N',
                        ipivot::Vector{BlasInt} = Vector{BlasInt}(undef,size(A,2)),
                        sparse::Real = 0.0,
                        pack::AbstractChar = 'N',
                        dmax = one($elty),
                        d = Vector{$elty}(undef, size(A,2)),
                        dl = Vector{$relty}(undef, size(A,2)),
                        dr = Vector{$relty}(undef, size(A,2))
                        )
            n = size(A,2)
            lda = size(A,1)
            m = lda # CHECKME: is it worth generalizing?

            (imode == 0) && throw(ArgumentError("imode = 0 needs explicit d,ds; not yet implemented"))

            iwork = Vector{BlasInt}(undef, max(m,n))
            info = Ref{BlasInt}(0)

#=
            println("zlatmr arg list:")
            println(
                  " m $m, n $n, dist $dist, iseed $iseed, sym $sym, d $d, imode $imode, rcond $rcond, dmax $dmax,
                  rsign $(rsign ? 'T' : 'F'), grade $grade, dl $dl, model $model, rcondl $rcondl,
                  dr $dr, moder $moder, rcondr $rcondr, pivtng $pivtng, ipivot $ipivot,
                  kl $kl, ku $ku, sparse $sparse, anorm $anorm, pack $pack, A, lda $lda")
=#
            ccall((@blasfunc($fname), liblapack), Cvoid,
                  (Ref{BlasInt}, # m
                   Ref{BlasInt}, # n
                   Ref{UInt8}, # dist
                   Ptr{BlasInt}, # iseed
                   Ref{UInt8}, # sym
                   Ptr{$elty}, # d(w)
                   Ref{BlasInt}, # mode
                   Ref{$relty}, # cond
                   Ref{$elty}, # dmax
                   Ref{UInt8}, # rsign
                   Ref{UInt8}, # grade
                   Ptr{$relty}, # dl(w)
                   Ref{BlasInt}, # model
                   Ref{$relty}, # condl
                   Ptr{$relty}, # dr(w)
                   Ref{BlasInt}, # moder
                   Ref{$relty}, # condr
                   Ref{UInt8}, # pivtng
                   Ptr{BlasInt}, # ipivot
                   Ref{BlasInt}, # kl
                   Ref{BlasInt}, # ku
                   Ref{$relty}, # sparse
                   Ref{$relty}, # anorm
                   Ref{UInt8}, # pack
                   Ptr{$elty}, # A
                   Ref{BlasInt}, # lda
                   Ptr{BlasInt}, # iwork(w)
                   Ptr{BlasInt} # info(w)
                   ),
                  m, n, dist, iseed, sym, d, imode, rcond, dmax,
                  rsign ? 'T' : 'F', grade, dl, model, rcondl,
                  dr, moder, rcondr, pivtng, ipivot,
                  kl, ku, sparse, anorm, pack, A, lda, iwork, info)
#            (info[] == 0) || error("info = $(info[]) from latmr")
            A, d, dl, dr, iseed
        end
    end
end

end # module
