# GenericSchur

<!-- ![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg) -->
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![GitHub CI Build Status](https://github.com/RalphAS/GenericSchur.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/RalphAS/GenericSchur.jl/actions/workflows/CI.yml)
[![codecov.io](http://codecov.io/github/RalphAS/GenericSchur.jl/coverage.svg?branch=master)](http://codecov.io/github/RalphAS/GenericSchur.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://RalphAS.github.io/GenericSchur.jl/dev)

## Eigen-analysis of matrices with generic floating-point element types in Julia

The Schur decomposition is the workhorse for eigensystem analysis of
dense matrices. The diagonal eigen-decomposition of normal
(especially Hermitian) matrices is an important special case,
but for non-normal matrices the Schur form is often more useful.

The purpose of this package is to extend the `schur!` and related
functions of the standard library to number types not handled by
LAPACK, such as `Complex{BigFloat}, Complex{Float128}` (from
Quadmath.jl), etc.  For these, the `schur!`, `eigvals!`, and `eigen!`
functions in the `LinearAlgebra` standard library are overloaded here,
and may be accessed through the usual `schur`, `eigvals`, and `eigen`
wrappers. For example:

```julia
A = BigFloat.(your_matrix_generator())
S = schur(A)
```
The result `S` is a `LinearAlgebra.Schur` object, with the properties `T`,
`Z=vectors`, and `values`.

The unexported `gschur` and `gschur!` functions are available for types
normally handled by the LAPACK wrappers in `LinearAlgebra`.

### Hermitian matrices

For these the Schur decomposition is a full diagonalization. This package provides
this for real symmetric and complex Hermitian matrices. Currently the QR iteration
and divide-and-conquer algorithms are implemented for the final reduction.

### Complex non-Hermitian matrices

For square matrices of complex element type,
this package provides a full Schur decomposition:
```julia
A::StridedMatrix{C} where {C <: Complex} == Z * T * adjoint(Z)
```
where `T` is upper-triangular and `Z` is unitary, both with the same element
type as `A`. (See below for real matrices.)

The algorithm is essentially the unblocked, serial, single-shift Francis (QR)
scheme used in the complex LAPACK routines. Balancing is also available.

### Eigenvectors

Right and left eigenvectors are available from Schur factorizations,
using

```julia
S = schur(A)
VR = eigvecs(S)
VL = eigvecs(S,left=true)
```

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
The result is in standard form, so
pair-blocks (and therefore rank-2 invariant subspaces) should be fully resolved.

One can convert a standard quasi-triangular real `Schur`
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
Tests to date suggest that behavior is similar
to the LAPACK routines on which the implementation is based.


## Generalized eigensystems

Methods for the generalized eigenvalue problem (matrix pencils) are provided,
as extensions of `schur!(A,B)` from LinearAlgebra.
The algorithms are translated from LAPACK, but this implementation has
had limited testing. (Note that it is easy to check the decomposition
of a particular case ex post facto.)

Corresponding functions for reordering (via `ordschur`) and
condition estimation (`eigvalscond`) are provided.
Tests to date suggest that behavior is similar to LAPACK.

Left and right eigenvectors of real and complex generalized problems are available with
`eigvecs(S::GeneralizedSchur; left::Bool)`.

These vectors currently have a peculiar normalization intended to be compatible with LAPACK
conventions.

## Acknowledgements

This package includes many translations from
[LAPACK](http://www.netlib.org/lapack/index.html) code, and
incorporates or elaborates a few methods from Andreas Noack's
[GenericLinearAlgebra.jl](http://github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl)
package.
