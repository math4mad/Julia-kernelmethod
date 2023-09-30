"""
from   https://juliaai.github.io/DataScienceTutorials.jl/isl/lab-9/index.html
https://discourse.julialang.org/t/how-do-i-specify-a-
-kernel-in-svm-using-libsvm-or-mlj/61962
"""

include("./src/memoize.jl")

import MLJ: fit!, predict

using  .Memoize
using CSV
using DataFrames
using KernelFunctions
using MLJ
using Plots
using PrettyPrinting
using Random





urls(str)="/Users/lunarcheung/Public/DataSets/clustering-datasets/$str.csv"
f(str)=urls(str)|>CSV.File|>DataFrame



"""
 "dart"    PolynomialKernel(; degree=5, c=2)  
 "dat2"    PolynomialKernel(; degree=5, c=3)
"""
pic="blob"


df=f(pic)

rows,cols=size(df)

 x1=df[1:2:rows,:x]
 x2=df[1:2:rows,:y]
 X=hcat(x1,x2)
 y=df[1:2:rows,:color]

 X = MLJ.table(X)
 y = categorical(y)



#test
 n1=n2=200
 xlow,xhigh=extrema(df[:,:x])
 ylow,yhigh=extrema(df[:,:y])
 tx = range(xlow,xhigh; length=n1)
 ty = range(ylow,yhigh; length=n2)
 x_test = mapreduce(collect, hcat, Iterators.product(tx, ty));
x_test=MLJ.table(x_test')


#define kernelmethods
k1 = PolynomialKernel(; degree=2, c=1)
k2 = SqExponentialKernel() ∘ ScaleTransform(1.5)
k3 = SqExponentialKernel() 
k4=  PolynomialKernel(; degree=2, c=1)
k5=  Matern52Kernel()
k6=GaussianKernel()
k7=RBFKernel()


@time  SVC = @load SVC pkg=LIBSVM

@time   svc_mdl = SVC(kernel=k4)

@time   svc = machine(svc_mdl, X, y)

@time fit!(svc);


ypred=predict(svc, x_test)

#plot
cat=df[:,:color]|>levels|>length
@time   contourf(tx,ty,ypred,levels=cat,color=cgrad(:redsblues),alpha=0.7)
@time   p1=scatter!(df[:,:x],df[:,:y],group=df[:,:color],label=false,ms=3,alpha=0.3)
#savefig(p1,"./$pic-svm.png")








