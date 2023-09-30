"""
Kung S.Y. - Kernel Methods and Machine Learning-CUP (2014).pdf page 85 example2.3   
"""

using   KernelFunctions, LinearAlgebra,Symbolics

xs=[1,2,0]
ys=[0,0,1]
zs=[1, -1]
ma=hcat(xs,ys)

# k=PolynomialKernel(; degree=2, c=1)  # K(x,y)=(1+(x*y))^2

# K = kernelmatrix(k, RowVecs(ma))

# k = SqExponentialKernel()
# x1 = rand(3)
# x2 = rand(3)
# k(x1, x2)



#  @variables   x₁,x₂,x₃,z₁,z₂,z₃,c,M
 
#  x=[x₁,x₂,x₃]
#  z=[z₁,z₂,z₃]

# expr=(x'*z+c)^M

# expr3=substitute(expr,Dict(M=>3,c=>1//2))

# #expand(order3)