"""
kmml Problem 1.8 (Supervised learning: over-determined system)
w=Xᵀ(XXᵀ)⁻¹y
"""

using LinearAlgebra

X=[1 1 -2 0;
   -2 2 0 0 ;
   1 0 1 -2
]

y=vec([1,1,-1,-1])

#w=(X'*inv(X*X'))*y
#X'inv(X*X')*y
w=inv(X'*X)*(X'*y)
