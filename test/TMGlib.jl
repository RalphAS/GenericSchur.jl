module TMGlib
# wrap some routines from the LAPACK test matrix generator library
export latme!, latmr!

using LinearAlgebra
using LinearAlgebra: BlasInt, checksquare
using LinearAlgebra.BLAS: @blasfunc

# CHECKME: routines linked here are included in many BLAS/LAPACK distributions
# (OpenBLAS, MKL), but do we need to allow for others?

const liblapack = Base.liblapack_name

const latm_modes = Dict(
    :specified => 0, :clustered_small => 1,
    :clustered_large => 2, :exponential => 3,
    :arithmetic => 4, :random_log => 5, :random => 6
)

function _check_iseed(iseed::Vector{T}) where {T <: Integer}
    if length(iseed) < 4
        throw(ArgumentError("iseed must have (at least) 4 elements"))
    end
    if any(iseed .< 0) || any(iseed .> 4095)
        throw(ArgumentError("iseed entries must be in 0:4095"))
    end
    isodd(iseed[4]) || throw(ArgumentError("iseed[4] must be odd"))
    return nothing
end

"""
construct a random non-symmetric square matrix with specified eigenvalues

`latme!(A::Matrix{<:BlasFloat}, anorm::Real, imode::Int, rcond::Real,
simrcond::Real = 1.0;
iseed::Vector{BlasInt} = [2518, 3899, 995, 397],
kl::Int = size(A,2), ku::Int = size(A,2),
dist::AbstractChar = 'D', rsign::Bool = true,
upper::Bool = true, simtrans::Bool = true,
simmode::Int = 4, d::Vector, ds::Vector)`

- `imode`: 1 - clustered small, 2 - clustered large, 3 - exponential,
     4 - arithmetic, 5 - random log, 6 - random (<0 for reversed order),
     0 - eigenvalues explicitly specified in `d`.
- `rcond`: ratio of extreme eigenvalue magnitudes.
- `anorm`: if positive, finally scale the matrix so its maximum-element-norm is `anorm`.
- `kl,ku`: lower and upper bandwidths
- If `upper`, fill the upper triangle (make result non-normal).
- If `rsign`, eigenvalues have random signs (real) or angles (complex).
- If `simtrans`, apply a similarity transform with singular values
      specified by `simmode, simrcond, ds`, similarly to `imode, rcond, d`.

- `dist`: character key for distribution to be sampled: `'U'` - uniform(0,1),
   `'S'` - symmetric uniform(-1,1), `'N'` - normal, `'D'` - uniform on complex unit disk.
"""
function latme! end

for (fname, elty, relty) in (
        (:zlatme_, :ComplexF64, :Float64),
        (:clatme_, :ComplexF32, :Float32),
    )
    @eval begin
        function latme!(
                A::Matrix{$elty}, anorm::Real, imode::Int, rcond::Real,
                simrcond::Real = 1.0;
                iseed::Vector{BlasInt} = BlasInt[2518, 3899, 995, 397],
                kl::Int = size(A, 2), ku::Int = size(A, 2),
                dist::AbstractChar = 'D', rsign::Bool = true,
                upper::Bool = true, simtrans::Bool = true,
                simmode::Int = 4,
                dmax::$elty = one($elty),
                d::Vector{$elty} = Vector{$elty}(undef, size(A, 2)),
                ds::Vector{$relty} = Vector{$relty}(undef, size(A, 2))
            )
            n = size(A, 2)
            lda = size(A, 1)

            if imode == 0
                (length(d) == n) || throw(ArgumentError("imode = 0 needs explicit d of length n"))
                if simtrans
                    (length(ds) == n) || throw(ArgumentError("imode = 0 needs explicit ds of length n"))
                end
            end
            (dist ∈ ['D', 'U', 'S', 'N']) || throw(ArgumentError("allowed values of dist are 'D','U','S', and 'N'"))

            _check_iseed(iseed)
            work = Vector{$elty}(undef, 3 * n)
            info = Ref{BlasInt}(0)

            ccall(
                (@blasfunc($fname), liblapack), Cvoid,
                (
                    Ref{BlasInt}, # n
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
                    Ptr{BlasInt}, # info(w)
                    Clong, Clong, Clong, Clong, # char lengths
                ),
                n, dist, iseed, d, imode, rcond, dmax,
                rsign ? 'T' : 'F', upper ? 'T' : 'F', simtrans ? 'T' : 'F',
                ds, simmode, simrcond, kl, ku, anorm, A, lda, work, info, 1, 1, 1, 1
            )
            (info[] == 0) || error("info = $(info[]) from latme!")
            return A, d, ds, iseed
        end
    end
end

# Note LAPACK routine has different arg list for the real versions:
for (fname, celty, elty) in (
        (:dlatme_, :ComplexF64, :Float64),
        (:slatme_, :ComplexF32, :Float32),
    )
    @eval begin
        function latme!(
                A::Matrix{$elty}, anorm::Real, imode::Int, rcond::Real,
                simrcond::Real = 1.0;
                iseed::Vector{BlasInt} = BlasInt[2518, 3899, 995, 397],
                kl::Int = size(A, 2), ku::Int = size(A, 2),
                dist::AbstractChar = 'S', rsign::Bool = true,
                upper::Bool = true, simtrans::Bool = true,
                simmode::Int = 4,
                dmax::$elty = one($elty),
                d = nothing,
                ds::Vector{$elty} = Vector{$elty}(undef, size(A, 2))
            )
            n = size(A, 2)
            lda = size(A, 1)

            d1 = Vector{$elty}(undef, size(A, 2))

            if imode == 0
                (d isa Vector) || throw(ArgumentError("imode = 0 needs vector d"))
                (length(d) == n) || throw(ArgumentError("imode = 0 needs explicit d of length n"))
                if simtrans
                    (length(ds) == n) || throw(ArgumentError("imode = 0 needs explicit ds of length n"))
                end
            end
            (dist ∈ ['U', 'S', 'N']) || throw(ArgumentError("allowed values of dist are 'U','S', and 'N'"))

            _check_iseed(iseed)
            work = Vector{$elty}(undef, 3 * n)
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

            if imode == 0
                if eltype(d) <: Real
                    ei = UInt8[' ']
                    d1 .= d[1:n]
                else
                    ei = Vector{UInt8}(undef, n)
                    ie = 1
                    id = 1
                    while id <= n
                        ei[ie] = 'R'
                        d1[ie] = real(d[id])
                        ie += 1
                        if imag(d[id]) != 0
                            if d[id + 1] != conj(d[id])
                                throw(ArgumentError("conjugate pairs in d must be adjacent"))
                            end
                            ei[ie] = 'I'
                            d1[ie] = imag(d[id])
                            ie += 1
                            id += 1
                        end
                        id += 1
                    end
                end
            else
                ei = UInt8[' ']
            end

            ccall(
                (@blasfunc($fname), liblapack), Cvoid,
                (
                    Ref{BlasInt}, # n
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
                    Ptr{BlasInt}, # info(w)
                    Clong, Clong, Clong, Clong, Clong, # char lengths
                ),
                n, dist, iseed, d1, imode, rcond, dmax, ei,
                rsign ? 'T' : 'F', upper ? 'T' : 'F', simtrans ? 'T' : 'F',
                ds, simmode, simrcond, kl, ku, anorm, A, lda, work, info,
                1, 1, 1, 1, 1
            )
            (info[] == 0) || error("info = $(info[]) from latme!")
            return A, d1, ds, iseed
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
                pack::AbstractChar = 'N',
                d::Vector, dl::Vector, dr::Vector
                )

- `imode`: 1 - clustered small, 2 - clustered large, 3 - exponential,
       4 - arithmetic, 5 - random log, 6 - random (<0 for reversed order),
       0 - eigenvalues explicitly specified in `d`.
- `sym`: `'S'` - symmetric, `'H'` - Hermitian, `'N'` - nonsymmetric
- `rcond`: ratio of extreme diagonal elements
- `kl,ku`: lower and upper bandwidths
- `sparse`: fraction to be nulled out
- `dist`: character key for distribution to be sampled: `'U'` - uniform(0,1),
   `'S'` - symmetric uniform(-1,1), `'N'` - normal, `'D'` - uniform on complex unit disk.
- `grade`: `'N'` - do not grade, otherwise a key for pre/postmultiplication with
      `Diagonal(dl), Diagonal(dr)`.
"""
function latmr! end

for (fname, elty, relty) in (
        (:zlatmr_, :ComplexF64, :Float64),
        (:clatmr_, :ComplexF32, :Float32),
        (:dlatmr_, :Float64, :Float64),
        (:slatmr_, :Float32, :Float32),
    )
    @eval begin
        function latmr!(
                A::Matrix{$elty}, anorm::Real, imode::Int, rcond::Real,
                rcondr::Real = 1.0, rcondl::Real = 1.0;
                sym::AbstractChar = 'N',
                iseed::Vector{BlasInt} = BlasInt[2518, 3899, 995, 397],
                kl::Int = size(A, 2), ku::Int = size(A, 2),
                dist::AbstractChar = $elty <: Complex ? 'D' : 'S',
                rsign::Bool = true,
                grade::AbstractChar = 'N', mode_l::Int = 1, mode_r::Int = 1,
                pivtng::AbstractChar = 'N',
                ipivot::Vector{BlasInt} = Vector{BlasInt}(undef, size(A, 2)),
                sparse::Real = 0.0,
                pack::AbstractChar = 'N',
                dmax = one($elty),
                d = Vector{$elty}(undef, size(A, 2)),
                dl = Vector{$relty}(undef, size(A, 2)),
                dr = Vector{$relty}(undef, size(A, 2))
            )
            n = size(A, 2)
            lda = size(A, 1)
            m = lda # CHECKME: is it worth generalizing?

            _check_iseed(iseed)

            iwork = Vector{BlasInt}(undef, max(m, n))
            info = Ref{BlasInt}(0)

            #=
            println("zlatmr arg list:")
            println(
                  " m $m, n $n, dist $dist, iseed $iseed, sym $sym, d $d, imode $imode, rcond $rcond, dmax $dmax,
                  rsign $(rsign ? 'T' : 'F'), grade $grade, dl $dl, mode_l $mode_l, rcondl $rcondl,
                  dr $dr, mode_r $mode_r, rcondr $rcondr, pivtng $pivtng, ipivot $ipivot,
                  kl $kl, ku $ku, sparse $sparse, anorm $anorm, pack $pack, A, lda $lda")
=#
            ccall(
                (@blasfunc($fname), liblapack), Cvoid,
                (
                    Ref{BlasInt}, # m
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
                    Ref{BlasInt}, # mode_l
                    Ref{$relty}, # condl
                    Ptr{$relty}, # dr(w)
                    Ref{BlasInt}, # mode_r
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
                    Ptr{BlasInt}, # info(w)
                    Clong, Clong, Clong, Clong, Clong, Clong,
                ),
                m, n, dist, iseed, sym, d, imode, rcond, dmax,
                rsign ? 'T' : 'F', grade, dl, mode_l, rcondl,
                dr, mode_r, rcondr, pivtng, ipivot,
                kl, ku, sparse, anorm, pack, A, lda, iwork, info,
                1, 1, 1, 1, 1, 1
            )
            (info[] == 0) || error("info = $(info[]) from latmr")
            return A, d, dl, dr, iseed
        end
    end
end

"""
construct sets of matrices satisfying a generalized Sylvester system

`A R - L B = C; D R - L E = F`

and diagonalization conditions.
"""
function latm5! end

for (fname, elty, relty) in (
        (:zlatm5_, :ComplexF64, :Float64),
        (:clatm5_, :ComplexF32, :Float32),
        (:dlatm5_, :Float64, :Float64),
        (:slatm5_, :Float32, :Float32),
    )
    @eval begin
        function latm5!(
                prtype::Integer, A::StridedArray{$elty}, B::StridedArray{$elty},
                C::StridedArray{$elty}, D::StridedArray{$elty}, E::StridedArray{$elty},
                F::StridedArray{$elty}, R::StridedArray{$elty}, L::StridedArray{$elty},
                alpha, qblocka, qblockb
            )
            m = checksquare(A)
            lda = m
            n = checksquare(B)
            ldb = n
            # FIXME: ensure D is mxm, E is nxn; C,F,R,L are mxn
            ldd, dle = m, n
            ldc, ldf, ldr, ldl = m, m, m, m
            return ccall(
                (@blasfunc($fname), liblapack), Cvoid,
                (
                    Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ref{$relty}, Ref{BlasInt}, Ref{BlasInt},
                ),
                prtype, m, n,
                A, max(1, stride(A, 2)), B, max(1, stride(B, 2)),
                C, max(1, stride(C, 2)), D, max(1, stride(D, 2)),
                E, max(1, stride(E, 2)), F, max(1, stride(F, 2)),
                R, max(1, stride(R, 2)), L, max(1, stride(L, 2)),
                alpha, qblocka, qblockb
            )
        end
    end
end

end # module
