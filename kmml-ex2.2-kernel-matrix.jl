"""

Kung S.Y. - Kernel Methods and Machine Learning-CUP (2014).pdf page 80 example2.2
"""


using Plots,KernelFunctions, LinearAlgebra

xs=[1, -1, 1,-1]
ys=[1,-1,-1, 1]
zs=[1, 1,0, 0]
ma=hcat(xs,ys)


k=PolynomialKernel(; degree=2, c=1)  # K(x,y)=(1+(x*y))^2
K = kernelmatrix(k, RowVecs(ma))

