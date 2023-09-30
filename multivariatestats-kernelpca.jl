using MultivariateStats,KernelFunctions, LinearAlgebra

xs=[1, -1, 1,-1]
ys=[1,-1,-1, 1]
zs=[1, 1,0, 0]
ma=hcat(xs,ys)


# struct MyKernel <: KernelFunctions.Kernel end
# (k::MyKernel)(x,y) = exp(-abs(x-y)^2)

#fit(KernelPCA, ma';)
