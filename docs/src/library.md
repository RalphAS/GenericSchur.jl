# API
Most of the user-facing methods are extensions of functions declared in
LinearAlgebra: `eigen!`, `schur!`, `ordschur!`, `eigvals!`.

Additional public functions are as follows.

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

## Non-piratical functions
If one opts out of type piracy, the implementations in `GenericSchur` are available
via the following functions. (Several are not currently exported, but *are*
considered public API.)

Mainly for testing purposes, one can also invoke the generic methods implemented
here for LAPACK-compatible matrix types with these.

### Schur decomposition
```@docs
GenericSchur.gschur!

GenericSchur.gschur

GenericSchur.ggschur!
```

### Eigenvectors
```@docs
GenericSchur.geigen!

GenericSchur.geigvecs
```

### Reordering a Schur factorization
```@docs
GenericSchur.gordschur!
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
