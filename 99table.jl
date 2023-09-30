using KernelFunctions
k=PolynomialKernel(;degree=1)
x=Vector(1:9)
kernelmatrix(k, x).|>Int

k(9,9)