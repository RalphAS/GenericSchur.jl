# GenericSchur

<!-- ![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg) -->
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.org/RalphAS/GenericSchur.jl.svg?branch=master)](https://travis-ci.org/RalphAS/GenericSchur.jl)
[![codecov.io](http://codecov.io/github/RalphAS/GenericSchur.jl/coverage.svg?branch=master)](http://codecov.io/github/RalphAS/GenericSchur.jl?branch=master)

## Schur decomposition for matrices of generic element types in Julia

This package provides a full Schur decomposition of complex square matrices:
```julia
A::StridedMatrix{C} where {C <: Complex} == Z * T * adjoint(Z)
```
where `T` is upper-triangular and `Z` is unitary, both with the same element
type as `A`.

The principal application is to generic number types such as `Complex{BigFloat}`.
For these, the `schur!` and `eigvals!` functions in the `LinearAlgebra`
standard library are overloaded, and may be accessed through the usual
`schur` and `eigvals` wrappers:

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

### Eigenvectors

Right eigenvectors are available from complex Schur factorizations, using

```julia
S = schur(A)
V = eigvecs(S)
```
The results are currently unreliable if the Frobenius norm of `A` is very
small or very large, so scale if necessary.

### Real decompositions

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

Eigenvectors are not currently available for the "real Schur" forms.

## Acknowledgements

This package incorporates or elaborates several methods from Andreas Noack's
[GenericLinearAlgebra.jl](http://github.com/AndreasNoack/GenericLinearAlgebra.jl) package,
and includes translations from [LAPACK](http://www.netlib.org/lapack/index.html) code.
