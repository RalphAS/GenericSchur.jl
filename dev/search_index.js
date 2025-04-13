var documenterSearchIndex = {"docs":
[{"location":"library/#API","page":"Library","title":"API","text":"","category":"section"},{"location":"library/","page":"Library","title":"Library","text":"Most of the user-facing methods are extensions of functions declared in LinearAlgebra: eigen!, schur!, ordschur!, eigvals!.","category":"page"},{"location":"library/","page":"Library","title":"Library","text":"Additional exported functions are as follows.","category":"page"},{"location":"library/#Balancing","page":"Library","title":"Balancing","text":"","category":"section"},{"location":"library/","page":"Library","title":"Library","text":"balance!","category":"page"},{"location":"library/#GenericSchur.balance!","page":"Library","title":"GenericSchur.balance!","text":"balance!(A; scale=true, permute=true) => Abal, B::Balancer\n\nBalance a square matrix so that various operations are more stable.\n\nIf permute, then row and column permutations are found such that Abal has the block form [T₁ X Y; 0 C Z; 0 0 T₂] where T₁ and T₂ are upper-triangular. If scale, then a diagonal similarity transform using powers of 2 is found such that the 1-norm of the C block (or the whole of Abal, if not permuting) is near unity. The transformations are encoded into B so that they can be inverted for eigenvectors, etc. Balancing typically improves the accuracy of eigen-analysis.\n\n\n\n\n\n","category":"function"},{"location":"library/#Converting-a-real-quasi-Schur-to-a-true-complex-Schur-object","page":"Library","title":"Converting a real quasi-Schur to a true complex Schur object","text":"","category":"section"},{"location":"library/","page":"Library","title":"Library","text":"triangularize","category":"page"},{"location":"library/#GenericSchur.triangularize","page":"Library","title":"GenericSchur.triangularize","text":"triangularize(S::Schur{T}) => Schur{complex{T})\n\nconvert a (standard-form) quasi-triangular real Schur factorization into a triangular complex Schur factorization.\n\n\n\n\n\n","category":"function"},{"location":"library/#Obtaining-eigenvectors-from-a-decomposition","page":"Library","title":"Obtaining eigenvectors from a decomposition","text":"","category":"section"},{"location":"library/","page":"Library","title":"Library","text":"LinearAlgebra.eigvecs(::Schur{Complex{T}}) where {T <: AbstractFloat}\n","category":"page"},{"location":"library/#LinearAlgebra.eigvecs-Union{Tuple{Schur{Complex{T}}}, Tuple{T}} where T<:AbstractFloat","page":"Library","title":"LinearAlgebra.eigvecs","text":"eigvecs(S::Schur{Complex{<:AbstractFloat}}; left=false) -> Matrix\n\nCompute right or left eigenvectors from a Schur decomposition. Eigenvectors are returned as columns of a matrix, ordered to match S.values. The returned eigenvectors have unit Euclidean norm, and the largest elements are real.\n\n\n\n\n\n","category":"method"},{"location":"library/#Eigenvalue-condition-numbers","page":"Library","title":"Eigenvalue condition numbers","text":"","category":"section"},{"location":"library/","page":"Library","title":"Library","text":"eigvalscond","category":"page"},{"location":"library/#GenericSchur.eigvalscond","page":"Library","title":"GenericSchur.eigvalscond","text":"eigvalscond(S::Schur,nsub::Integer) => Real\n\nEstimate the reciprocal of the condition number of the nsub leading eigenvalues of S. (Use ordschur to move a subspace of interest to the front of S.)\n\nSee the LAPACK User's Guide for details of interpretation.\n\n\n\n\n\neigvalscond(S::GeneralizedSchur, nsub) => pl, pr\n\ncompute approx. reciprocal norms of projectors on left/right subspaces associated w/ leading nsub×nsub block of S. Use ordschur to select eigenvalues of interest.\n\nAn approximate bound on avg. absolute error of associated eigenvalues is ϵ * norm(vcat(A,B)) / pl. See LAPACK documentation for further details.\n\n\n\n\n\n","category":"function"},{"location":"library/#Subspace-conditioning","page":"Library","title":"Subspace conditioning","text":"","category":"section"},{"location":"library/","page":"Library","title":"Library","text":"subspacesep","category":"page"},{"location":"library/#GenericSchur.subspacesep","page":"Library","title":"GenericSchur.subspacesep","text":"subspacesep(S::Schur,nsub::Integer) => Real\n\nEstimate the reciprocal condition of the separation angle for the invariant subspace corresponding to the leading block of size nsub of a Schur decomposition. (Use ordschur to move a subspace of interest to the front of S.)\n\nSee the LAPACK User's Guide for details of interpretation.\n\n\n\n\n\n","category":"function"},{"location":"library/#Circumventing-normal-dispatch","page":"Library","title":"Circumventing normal dispatch","text":"","category":"section"},{"location":"library/","page":"Library","title":"Library","text":"Mainly for testing purposes, one can invoke the generic methods implemented here for LAPACK-compatible matrix types with these methods.","category":"page"},{"location":"library/","page":"Library","title":"Library","text":"GenericSchur.gschur!\n\nGenericSchur.gschur\n\nGenericSchur.geigen!","category":"page"},{"location":"library/#GenericSchur.gschur!","page":"Library","title":"GenericSchur.gschur!","text":"gschur!(A::StridedMatrix) -> F::Schur\n\nDestructive version of gschur (q.v.).\n\n\n\n\n\ngschur!(H::Hessenberg, Z) -> F::Schur\n\nCompute the Schur decomposition of a Hessenberg matrix.  Subdiagonals of H must be real. If Z is provided, it is updated with the unitary transformations of the decomposition.\n\n\n\n\n\n","category":"function"},{"location":"library/#GenericSchur.gschur","page":"Library","title":"GenericSchur.gschur","text":"gschur(A::StridedMatrix) -> F::Schur\n\nComputes the Schur factorization of matrix A using a generic implementation. See LinearAlgebra.schur for usage.\n\n\n\n\n\n","category":"function"},{"location":"library/#GenericSchur.geigen!","page":"Library","title":"GenericSchur.geigen!","text":"geigen!(A, alg=QRIteration(); sortby=eigsortby) -> E::Eigen\n\nComputes the eigen-decomposition of a Hermitian (or real symmetric) matrix A using a generic implementation. Currently alg may be QRIteration() or DivideAndConquer(). Otherwise, similar to LinearAlgebra.eigen!.\n\n\n\n\n\n","category":"function"},{"location":"library/#Locally-defined-exceptions","page":"Library","title":"Locally defined exceptions","text":"","category":"section"},{"location":"library/","page":"Library","title":"Library","text":"Note: these are currently peculiar to this package but replacement by upstream analogues will not be considered breaking.","category":"page"},{"location":"library/","page":"Library","title":"Library","text":"GenericSchur.IllConditionException\n\nGenericSchur.UnconvergedException","category":"page"},{"location":"library/#GenericSchur.IllConditionException","page":"Library","title":"GenericSchur.IllConditionException","text":"IllConditionException\n\nException thrown when argument matrix or matrices are too ill-conditioned for the requested operation. The index field may indicate the block where near-singularity was detected.\n\n\n\n\n\n","category":"type"},{"location":"library/#GenericSchur.UnconvergedException","page":"Library","title":"GenericSchur.UnconvergedException","text":"UnconvergedException\n\nException thrown when an iterative algorithm does not converge within the allowed number of steps.\n\n\n\n\n\n","category":"type"},{"location":"#GenericSchur","page":"Overview","title":"GenericSchur","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"GenericSchur is a Julia package for eigen-analysis of dense real and complex matrices of generic numeric type, i.e. those not supported by the LAPACK-based routines in the standard LinearAlgebra library. Examples are BigFloat, Float128, and DoubleFloat.","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"The schur!, eigvals!, and eigen! functions in the LinearAlgebra standard library are overloaded here, and may be accessed through the usual schur, eigvals, and eigen wrappers.","category":"page"},{"location":"#Complex-Schur-decomposition","page":"Overview","title":"Complex Schur decomposition","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"The Schur decomposition is the workhorse for eigensystem analysis of dense non-symmetric matrices.","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"This package provides a full Schur decomposition of complex square matrices:","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"A::StridedMatrix{C} where {C <: Complex} == Z * T * adjoint(Z)","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"where T is upper-triangular and Z is unitary, both with the same element type as A.","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"A = your_matrix_generator() .+ 0im # in case you start with a real matrix\nS = schur(A)","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"The result S is a LinearAlgebra.Schur object, with the properties T, Z=vectors, and values.","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"The unexported gschur and gschur! functions are available for types normally handled by the LAPACK wrappers in LinearAlgebra.","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"The algorithm is essentially the unblocked, serial, single-shift Francis (QR) scheme used in the complex LAPACK routines. Scaling is enabled for balancing, but currently permutation (which would reduce the work) is not.","category":"page"},{"location":"#Hermitian-matrices","page":"Overview","title":"Hermitian matrices","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"The Schur decomposition of a Hermitian matrix is identical to diagonalization, i.e. the upper-triangular factor is diagonal. This package provides such decompositions for real-symmetric and complex-Hermitian matrices via eigen! etc. In these cases, an alg argument may be used to select a LinearAlgebra.QRIteration or LinearAlgebra.DivideAndConquer algorithm. The divide-and-conquer scheme takes a long time to compile, but execution is significantly faster than QR iteration for large matrices.","category":"page"},{"location":"#Real-decompositions","page":"Overview","title":"Real decompositions","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"A quasi-triangular \"real Schur\" decomposition of real matrices is also provided:","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"A::StridedMatrix{R} where {R <: Real} == Z * T * transpose(Z)","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"where T is quasi-upper-triangular and Z is orthogonal, both with the same element type as A.  This is what you get by invoking the above-mentioned functions with matrix arguments whose element type T <: Real. By default, the result is in standard form, so pair-blocks (and therefore rank-2 invariant subspaces) should be fully resolved. (This differs from the original version in GenericLinearAlgebra.jl.)","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"If the optional keyword standardized is set to false in gschur, a non-standard (but less expensive) form is produced.","category":"page"},{"location":"#Eigenvectors","page":"Overview","title":"Eigenvectors","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"Right and left eigenvectors are available from complex Schur factorizations, using","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"S = schur(A)\nVR = eigvecs(S)\nVL = eigvecs(S,left=true)","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"The results are currently unreliable if the Frobenius norm of A is very small or very large, so scale if necessary (see Balancing, below).","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"Eigenvectors are not currently available for the \"real Schur\" forms. But don't despair; one can convert a standard quasi-triangular real Schur into a complex Schur with the triangularize function provided here.","category":"page"},{"location":"#Balancing","page":"Overview","title":"Balancing","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"The accuracy of eigenvalues and eigenvectors may be improved for some matrices by use of a similarity transform which reduces the matrix norm.  This is done by default in the eigen! method, and may also be handled explicitly via the balance! function provided here:","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"Ab, B = balance!(copy(A))\nS = schur(Ab)\nv = eigvecs(S)\nlmul!(B, v) # to get the eigenvectors of A","category":"page"},{"location":"","page":"Overview","title":"Overview","text":"More details are in the function docstring. Although the balancing function also does permutations to isolate trivial subspaces, the Schur routines do not yet exploit this opportunity for reduced workload.","category":"page"},{"location":"#Generalized-problems","page":"Overview","title":"Generalized problems","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"The generalized Schur decomposition is provided for complex element types. Eigenvectors are also available, via eigvecs(S::GeneralizedSchur).","category":"page"},{"location":"#Acknowledgements","page":"Overview","title":"Acknowledgements","text":"","category":"section"},{"location":"","page":"Overview","title":"Overview","text":"This package includes translations from LAPACK code, and incorporates or elaborates several methods from Andreas Noack's GenericLinearAlgebra.jl package.","category":"page"}]
}
