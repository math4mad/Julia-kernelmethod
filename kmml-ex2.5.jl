"""
Kung S.Y. - Kernel Methods and Machine Learning-CUP (2014).pdf page 90 example2.5  
"""

using KernelFunctions
using LinearAlgebra
using SymbolicUtils

@syms x1::Vector x2::Vector A::Matrix

xs=[1,2,0]
ys=[0,0,1]
zs=[1, -1]
ma=hcat(xs,ys)

A=hcat(x1,x2)
 
k=PolynomialKernel(; degree=2, c=1) 

# kk(x::Vector)=[k(x,i) for i in eachcol(ma)]

K = kernelmatrix(k, ColVecs(ma))
a=(inv(K)*zs).|>(d->round(d,digits=3))



function f(x::Vector)
   return [k(xs,x) k(ys,x)]*a
end


f([1,2,3])