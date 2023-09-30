"""

Kung S.Y. - Kernel Methods and Machine Learning-CUP (2014).pdf page 128
"""


using Plots,KernelFunctions, LinearAlgebra

xs=[1, -1, 1,-1]
ys=[1,-1,-1, 1]
zs=[1, 1,0, 0]
ma=hcat(xs,ys)


#k=GaussianKernel()
struct MyKernel <: KernelFunctions.Kernel end
(k::MyKernel)(x,y) = exp(-abs(x-y)^2)

K = kernelmatrix(k, RowVecs(ma))

decomp=eigen(K)


# struct MyKernel <: KernelFunctions.Kernel end

# # (::Test)(x,y) = (1+x*y)^2
# (k::MyKernel)(x,y) = (1+(x*y))^2

# k = MyKernel()
# #X = rand(10, 2)
# matrix=[k(xs[i],ys[i]) for i in  1:4]