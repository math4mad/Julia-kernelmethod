"""
kmml page 70
"""

using KernelFunctions,Distances

x=1.0
y=10.0
#=========define sigmoid kernel function =============#
struct SigmoidKernel <: KernelFunctions.Kernel end
(::SigmoidKernel)(x, y) =1/(1+ℯ^(-x*y))
#KernelFunctions.metric(::SigmoidKernel) = SqEuclidean()
#======================================================#  

k1=PolynomialKernel(;degree=1,c=0) 
k2=PolynomialKernel(;degree=2,c=1.0) #define  polynomial kernel
k3=SqExponentialKernel()
k4=SigmoidKernel()  # 这是一个反例,不能获取欧式距离

 

dist_with_kernel=(;k)->(x,y)->sqrt(k(x,x)+k(y,y)-2*k(x,y))
euclidean_dist=dist_with_kernel(;k=k1)
polynomialKernel_dist=dist_with_kernel(;k=k2)
gaussian_dist=dist_with_kernel(;k=k3)
sigmoid_dist=dist_with_kernel(;k=k4)
euclidean_dist(x,y)
polynomialKernel_dist(x,y)
gaussian_dist(x,y)
#sigmoid_dist(x,y)  # sigmoid结果为复数, 不满足柯西不等式, 所以不能用于机器学习

