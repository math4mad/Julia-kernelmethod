"""
kmml page 88  example2.4 
"""

using  KernelFunctions, LinearAlgebra,Symbolics

@variables u,v,X
X=[u,v]

tm=[ 0 1 -1  1 -1  1  2;
     0 1 -1 -1  1  0  0
]

k=PolynomialKernel(; degree=2, c=1) 

kk(x::Vector)=[k(x,i) for i in eachcol(tm)]

kk(X)  #symbolics

kk([1,2])  #vector