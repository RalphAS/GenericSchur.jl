# API
Most of the user-facing methods are extensions of functions declared in
LinearAlgebra: `eigen!`, `schur!`, `ordschur!`, `eigvals!`.

Additional exported functions are as follows.

## Balancing

```@docs
balance!
```

## Converting a real quasi-Schur to a true complex Schur object

```@docs
triangularize
```

## Obtaining eigenvectors from a decomposition

```@docs
LinearAlgebra.eigvecs(::Schur{Complex{T}}) where {T <: AbstractFloat}

```

## Eigenvalue condition numbers

```@docs
eigvalscond
```

## Subspace conditioning

```@docs
subspacesep
```

## Circumventing normal dispatch
Mainly for testing purposes, one can invoke the generic methods implemented
here for LAPACK-compatible matrix types with these methods.
```@docs
GenericSchur.gschur!

GenericSchur.gschur

GenericSchur.geigen!
```

## Locally defined exceptions
Note: these are currently peculiar to this package but replacement by upstream analogues
will not be considered breaking.
```@docs
GenericSchur.IllConditionException

GenericSchur.UnconvergedException
```

## Preference handling
```@docs
GenericSchur.set_piracy!
```
