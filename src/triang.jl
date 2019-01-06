# This file is part of GenericSchur.jl, released under the MIT "Expat" license

"""
    triangularize(S::Schur{T}) => Schur{complex{T})

convert a (standard-form) quasi-triangular real Schur factorization into a
triangular complex Schur factorization.
"""
function triangularize(S::Schur{Ty}) where {Ty <: Real}
    CT = complex(Ty)
    Tr = S.T
    T = CT.(Tr)
    Z = CT.(S.Z)
    n = size(T,1)
    for j=n:-1:2
        if Tr[j,j-1] != zero(CT)
            # We want a unitary similarity transform from
            # ┌   ┐      ┌     ┐
            # │a b│      │w₁  x│
            # │c a│ into │0  w₂│ where bc < 0 (a,b,c real)
            # └   ┘      └     ┘
            # If we write it as
            # ┌     ┐
            # │u  v'│
            # │-v u'│
            # └     ┘
            # and make the Ansatz that u is real (so v is imaginary),
            # we arrive at:
            # θ = atan(sqrt(-Tr[j,j-1]/Tr[j-1,j]))
            # s = sin(θ)
            # c = cos(θ)
            s = sqrt(abs(Tr[j,j-1]))
            c = sqrt(abs(Tr[j-1,j]))
            r = hypot(s,c)
            G = Givens(j-1,j,complex(c/r),-im*(s/r))
            lmul!(G,T)
            rmul!(T,G')
            rmul!(Z,G')
        end
    end
    triu!(T)
    Schur(T,Z,diag(T))
end

