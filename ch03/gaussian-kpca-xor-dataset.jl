using LinearAlgebra,KernelFunctions,Plots


xs=[1, -1, 1,-1]
ys=[1,-1,-1, 1]
zs=[1, 1,0, 0]

matr=[0 0 1 1;
      0 1 0  1
]

matr2=hcat(xs,ys)'


k=ExponentialKernel()

K = kernelmatrix(k, ColVecs(matr2))

(Λ,U)=eigen(K,sortby=-)


# kk(x::Vector)=[k(x,i) for i in eachcol(matr)]

# fk(;vec)=((diagm(Λ))^(-1/2))*U*kk(vec)


# fk(vec=matr[:,1])



