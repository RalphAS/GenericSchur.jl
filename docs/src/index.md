# GenericSchur

GenericSchur is a Julia package for eigen-analysis of non-symmetric
real and complex matrices of generic numeric type, i.e. those
not supported by the LAPACK-based routines in the standard LinearAlgebra
library. Examples are `BigFloat`, `Float128`, and `DoubleFloat`.

## Complex Schur decomposition

The Schur decomposition is the workhorse for eigensystem analysis of
dense non-symmetric matrices.

This package provides a full Schur decomposition of complex square matrices:
```julia
A::StridedMatrix{C} where {C <: Complex} == Z * T * adjoint(Z)
```
where `T` is upper-triangular and `Z` is unitary, both with the same element
type as `A`.

the `schur!`, `eigvals!`, and `eigen!` functions in the `LinearAlgebra`
standard library are overloaded here, and may be accessed through the usual
`schur`, `eigvals`, and `eigen` wrappers:

```julia
A = your_matrix_generator() + 0im # in case you start with a real matrix
S = schur(A)
```
The result `S` is a `LinearAlgebra.Schur` object, with the properties `T`,
`Z=vectors`, and `values`.

The unexported `gschur` and `gschur!` functions are available for types
normally handled by the LAPACK wrappers in `LinearAlgebra`.

The algorithm is essentially the unblocked, serial, single-shift Francis (QR)
scheme used in the complex LAPACK routines. Scaling is enabled for balancing,
but not permutation (which would reduce the work).

## Real decompositions

A quasi-triangular "real Schur" decomposition of real matrices is also
provided:
```julia
A::StridedMatrix{R} where {R <: Real} == Z * T * transpose(Z)
```
where `T` is quasi-upper-triangular and `Z` is orthogonal, both with the
same element type as `A`.  This is what you get by invoking the above-mentioned
functions with matrix arguments whose element type `T <: Real`.
By default, the result is in standard form, so
pair-blocks (and therefore rank-2 invariant subspaces) should be fully resolved.
(This differs from the original version in GenericLinearAlgebra.jl.)

If the optional keyword `standardized` is set to `false` in `gschur`, a
non-standard (but less expensive) form is produced.

## Eigenvectors

Right and left eigenvectors are available from complex Schur factorizations,
using

```julia
S = schur(A)
VR = eigvecs(S)
VL = eigvecs(S,left=true)
```
The results are currently unreliable if the Frobenius norm of `A` is very
small or very large, so scale if necessary (see Balancing, below).

Eigenvectors are not currently available for the "real Schur" forms.
But don't despair; one can convert a standard quasi-triangular real `Schur`
into a complex `Schur` with the `triangularize` function provided here.

## Balancing

The accuracy of eigenvalues and eigenvectors may be improved for some
matrices by use of a similarity transform which reduces the matrix
norm.  This is done by default in the `eigen!` method, and may also be
handled explicitly via the `balance!` function provided here:
```julia
Ab, B = balance!(copy(A))
S = schur(Ab)
v = eigvecs(S)
lmul!(B, v) # to get the eigenvectors of A
```
More details are in the function docstring. Although the balancing function
also does permutations to isolate trivial subspaces, the Schur routines do not
yet exploit this opportunity for reduced workload.

## Generalized problems

The generalized Schur decomposition is provided for complex element types.
Eigenvectors are also available, via `eigvecs(S::GeneralizedSchur)`.

## Acknowledgements

This package includes translations from [LAPACK](http://www.netlib.org/lapack/index.html)
code, and incorporates or elaborates several methods from Andreas Noack's
[GenericLinearAlgebra.jl](http://github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl)
package.



