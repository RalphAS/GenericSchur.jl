# Pirated methods
# This is the main public interface of the package for most users.
# Non-mutating Wrappers like `schur` and `eigvals` should just work.

# Some other stdlib functions (e.g., lmul!) are extended, but either non-piratically
# or with explicit qualification.

function LinearAlgebra.schur!(A::StridedMatrix{T}; kwargs...) where {T <: STypes}
    return gschur!(A; kwargs...)
end
function LinearAlgebra.schur!(
        A::StridedMatrix{T}, B::StridedMatrix{T}; kwargs...
    ) where {T <: STypes}
    return ggschur!(A, B; kwargs...)
end

function LinearAlgebra.eigvals!(
        A::StridedMatrix{T};
        sortby::Union{Function, Nothing} = eigsortby, kwargs...
    ) where {T <: STypes}
    S = gschur!(A; wantZ = false, kwargs...)
    if sortby !== nothing
        return sorteig!(S.values, sortby)
    else
        return S.values
    end
end

# This is probably the best we can do unless LinearAlgebra coöperates
"""
    eigvecs(S::Schur{Complex{<:AbstractFloat}}; left=false) -> Matrix

Compute right or left eigenvectors from a Schur decomposition.
Eigenvectors are returned as columns of a matrix, ordered to match `S.values`.
The returned eigenvectors have unit Euclidean norm, and the largest
elements are real.
"""
function LinearAlgebra.eigvecs(
        S::Schur{Complex{T}}; left::Bool = false
    ) where {T <: AbstractFloat}
    if left
        v = _gleigvecs!(S.T, S.Z)
    else
        v = _geigvecs!(S.T, S.Z)
    end
    _enormalize!(v)
    return v
end

function LinearAlgebra.eigvecs(S::GeneralizedSchur{T}; left::Bool = false
) where {T <: STypes}
    if left
        v = _gleigvecs(S.S, S.T, S.Q)
    else
        v = _geigvecs(S.S, S.T, S.Z)
    end
    # CHECKME: Euclidean norm differs from LAPACK, so wait for upstream.
    # _enormalize!(v)
    return v
end

function LinearAlgebra.eigen!(
        A::StridedMatrix{T}; permute::Bool = true, scale::Bool = true,
        sortby::Union{Function, Nothing} = eigsortby, kwargs...
    ) where {T <: STypes}
    n = checksquare(A)
    if permute || scale
        A, B = balance!(A, scale = scale, permute = permute)
    end
    if T <: Real
        S = triangularize(schur(A))
        VT = Complex{T}
    else
        S = schur(A)
        VT = T
    end
    if isempty(kwargs)
        v = _geigvecs!(S.T, S.Z)
        if permute || scale
            lmul!(B, v)
        end
        _enormalize!(v)
        if sortby !== nothing
            return LinearAlgebra.Eigen(sorteig!(S.values, v, sortby)...)
        else
            return LinearAlgebra.Eigen(S.values, v)
        end
    end

    # recip. cond. nr. needs both sets of eigvecs
    # but compute it for BALANCED forms for consistency
    do_vr = get(kwargs, :jvr, true) || get(kwargs, :jce, false)
    if do_vr
        v = _geigvecs!(S.T, S.Z)
    else
        v = zeros(VT, 0, 0)
    end
    do_vl = get(kwargs, :jvl, false) || get(kwargs, :jce, false)
    if do_vl
        vl = _gleigvecs!(S.T, S.Z)
    else
        vl = zeros(VT, 0, 0)
    end

    do_econd = get(kwargs, :jce, false)
    rconde = zeros(real(T), do_econd ? n : 0)
    if do_econd
        for j in 1:n
            vlj = view(vl, :, j)
            vrj = view(v, :, j)
            rconde[j] = abs(dot(vlj, vrj)) / norm(vlj) / norm(vrj)
        end
        if !get(kwargs, :jvr, true)
            v = zeros(VT, 0, 0)
        end
        if !get(kwargs, :jvl, true)
            vl = zeros(VT, 0, 0)
        end
    end

    if get(kwargs, :jvr, true)
        if permute || scale
            lmul!(B, v)
        end
        _enormalize!(v)
    end
    if get(kwargs, :jvl, true)
        if permute || scale
            ldiv!(B, vl)
        end
        _enormalize!(vl)
    end

    do_vcond = get(kwargs, :jcv, false)
    rcondv = zeros(real(T), do_vcond ? n : 0)
    if do_vcond
        # warning: this is very expensive
        Ttmp = similar(S.T)
        sel = falses(n)
        for j in 1:n
            copyto!(Ttmp, S.T)
            Stmp = Schur(Ttmp, similar(S.Z, 0, 0), copy(S.values))
            sel[j] = true
            ordschur!(Stmp, sel)
            sel[j] = false
            if do_econd
                rconde[j] = eigvalscond(Stmp, 1)
            end
            if do_vcond
                rcondv[j] = subspacesep(Stmp, 1)
            end
        end
    end
    if sortby !== nothing
        return LinearAlgebra.Eigen(sorteig!(S.values, v, sortby, vl, rconde, rcondv)...)
    else
        return LinearAlgebra.Eigen(S.values, v, vl, false, rconde, rcondv)
    end
end

function LinearAlgebra.eigen!(
        A::RealHermSymComplexHerm{<:STypes, <:StridedMatrix};
        alg::Algorithm = QRIteration(),
        kwargs...
    )
    return geigen!(A, alg; kwargs...)
end

function LinearAlgebra.eigvals!(
        A::RealHermSymComplexHerm{<:STypes, <:StridedMatrix};
        alg::Algorithm = QRIteration(),
        kwargs...
    )
    return geigvals!(A, alg; kwargs...)
end

function LinearAlgebra.eigen!(
        A::SymTridiagonal{<:AbstractFloat};
        alg::Algorithm = QRIteration(),
        kwargs...
    )
    return geigen!(A, alg; kwargs...)
end

function LinearAlgebra.eigvals!(
        A::SymTridiagonal{<:AbstractFloat};
        alg::Algorithm = QRIteration(),
        kwargs...
    )
    return geigvals!(A, alg; kwargs...)
end

# Some variants should be defined here for logical coherence, but not until
# we've registered nontrivial implementations.

# stdlib only provides not-in-place wrappers for Ty <: BlasFloat
# We are only claiming extension to STypes here
function LinearAlgebra._ordschur(
        T::StridedMatrix{Ty},
        Z::StridedMatrix{Ty},
        select::Union{Vector{Bool}, BitVector}
    ) where {Ty <: STypes}
    return _ordschur!(copy(T), copy(Z), select)
end

function LinearAlgebra._ordschur(
        S::StridedMatrix{Ty}, T::StridedMatrix{Ty},
        Q::StridedMatrix{Ty}, Z::StridedMatrix{Ty},
        select::Union{Vector{Bool}, BitVector}
    ) where {Ty <: STypes}
    return _ordschur!(copy(S), copy(T), copy(Q), copy(Z), select)
end

function LinearAlgebra._ordschur!(
        T::StridedMatrix{Ty},
        Z::StridedMatrix{Ty},
        select::Union{Vector{Bool}, BitVector}
    ) where {Ty <: STypes}
    return gordschur!(T, Z, select)
end

function LinearAlgebra._ordschur!(
        S::StridedMatrix{Ty}, T::StridedMatrix{Ty},
        Q::StridedMatrix{Ty}, Z::StridedMatrix{Ty},
        select::Union{Vector{Bool}, BitVector}
    ) where {Ty <: STypes}
    return gordschur!(S, T, Q, Z, select)
end

# CHECKME: is there a good reason to include these?
LinearAlgebra.hessenberg!(A::StridedMatrix{<:STypes}) = _hessenberg!(A)

using LinearAlgebra: RealHermSymComplexHerm
# this should be S <: StridedMatrix, but for stdlib foolishness
function LinearAlgebra.hessenberg!(Ah::RealHermSymComplexHerm{<:STypes, <:AbstractMatrix})
    return _hehessenberg!(Hermitian(Ah), Val(Symbol(Ah.uplo)))
end
