# GenericSchur

<!-- ![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg) -->
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![GitHub CI Build Status](https://github.com/RalphAS/GenericSchur.jl/workflows/CI/badge.svg)](https://github.com/RalphAS/GenericSchur.jl/actions)
[![codecov.io](http://codecov.io/github/RalphAS/GenericSchur.jl/coverage.svg?branch=master)](http://codecov.io/github/RalphAS/GenericSchur.jl?branch=master)

## Schur decomposition for matrices of generic element types in Julia

The Schur decomposition is the workhorse for eigensystem analysis of
dense non-symmetric matrices.

This package provides a full Schur decomposition of complex square matrices:
```julia
A::StridedMatrix{C} where {C <: Complex} == Z * T * adjoint(Z)
```
where `T` is upper-triangular and `Z` is unitary, both with the same element
type as `A`. (See below for real matrices.)

The principal application is to number types not handled by LAPACK,
such as `Complex{BigFloat}, Complex{Float128}` (from Quadmath.jl), etc.
For these, the `schur!`, `eigvals!`, and `eigen!` functions in the `LinearAlgebra`
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
scheme used in the complex LAPACK routines. Balancing is also available.

### Eigenvectors

Right and left eigenvectors are available from complex Schur factorizations,
using

```julia
S = schur(A)
VR = eigvecs(S)
VL = eigvecs(S,left=true)
```

The results are currently unreliable if the Frobenius norm of `A` is very
small or very large, so scale if necessary.

#### Normalization

As of v0.4, eigenvectors as returned from our `eigen` and `eigvecs` methods
for the standard problem have unit Euclidean norm. This accords with the
current (undocumented) behavior of `LinearAlgebra` methods. (Previously a
convention based on low-level LAPACK routines was used here.)

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

## Subspaces, condition, and all that.

Methods for reordering a Schur decomposition (`ordschur`) and computing
condition numbers (`eigvalscond`) and subspace separation (`subspacesep`)
are provided.
Tests to date suggest that behavior is analogous
to the LAPACK routines on which the implementation is based.


## Generalized eigensystems

Methods for the generalized eigenvalue problem (matrix pencils) with
`Complex` element types are available as of release 0.3.0;
in particular, extension of `schur(A,B)` from LinearAlgebra.
The algorithms are translated from LAPACK, but this implementation has
had limited testing. (Note that it is easy to check the decomposition
of a particular case ex post facto.)

Corresponding functions for reordering and condition
estimation are included. Tests to date suggest that behavior is analogous
to LAPACK.

Right eigenvectors of generalized problems are available with
`V = eigvecs(S::GeneralizedSchur{<:Complex})`. Column `j` of `V` satisfies
`S.beta[j] * A * v ≈ S.alpha[j] * B * v`.
These currently have a peculiar norm intended to be compatible with LAPACK
conventions.

## Acknowledgements

This package incorporates or elaborates several methods from Andreas Noack's
[GenericLinearAlgebra.jl](http://github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl) package,
and includes translations from [LAPACK](http://www.netlib.org/lapack/index.html) code.
