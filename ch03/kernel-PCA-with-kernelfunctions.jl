"""
"""

using KernelFunctions,MultivariateStats,LinearAlgebra,Symbolics

#@variables u,v,X
#X=[u,v]

xs=[1, -1, 1,-1]
ys=[1,-1,-1, 1]
zs=[1, 1,0, 0]

matr=[1 -1 1 -1;
      1 -1 -1 1
]

k=PolynomialKernel(; degree=2, c=1)
#k=SqExponentialKernel()

K = kernelmatrix(k, ColVecs(matr))

Λ,U=eigen(K,sortby=-)

kk(x::Vector)=[k(x,i) for i in eachcol(matr)]

#kexpr=kk(X)
#kexpr=kk([x,y])
#sfunc=(build_function(inv(diagm(Λ))*U*kexpr,[x,y]))|>eval

#fk(;vec)=(Λ^(-1/2))*U*kk(vec)
#res=[fk(vec=v) for v in eachcol(matr)]
#fk(vec=[1,0.1])


# 映射到 feature 空间的向量

fk(;vec)=((diagm(Λ))^(-1/2))*U*kk(vec)

fk(vec=matr[:,1])

