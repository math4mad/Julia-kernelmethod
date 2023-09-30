"""
kmml Problem 1.8 (Supervised learning: over-determined system)
w=Xᵀ(XXᵀ)⁻¹y
"""

using LinearAlgebra

X=[0.5 0.5;
   -2 2 ;
   1 -1
]
y=vec([1,-1])

#w=(X'*inv(X*X'))*y]
#X'inv(X*X')*y
w=inv(X'*X)*X'y
