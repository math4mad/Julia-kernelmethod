"""
from   https://juliaai.github.io/DataScienceTutorials.jl/isl/lab-9/index.html
https://discourse.julialang.org/t/how-do-i-specify-a-
-kernel-in-svm-using-libsvm-or-mlj/61962
"""

import MLJ: fit!, predict
using CSV
using DataFrames
using KernelFunctions
using MLJ
using Plots
using PrettyPrinting
using Random

#define kernelmethods
k1=PolynomialKernel(; degree=2, c=1)
k2 = SqExponentialKernel() ∘ ScaleTransform(1.5)
k3 = SqExponentialKernel() 
k4=PolynomialKernel(; degree=3, c=1)

urls(str)="/Users/lunarcheung/Public/DataSets/clustering-datasets/$str.csv"
f(str)=urls(str)|>CSV.File|>DataFrame


df=f("basic3")

x1=df[1:2:1000,:x]
x2=df[1:2:1000,:y]
X=hcat(x1,x2)
y=df[1:2:1000,:color]

#test1
xt1=df[1:2:4000,:x]
xt2=df[1:2:4000,:y]
Xt1=hcat(xt1,xt2)
yt1=df[1:2:4000,:color]

#test2

tr = range(35,550; length=150)
x_test = mapreduce(collect, hcat, Iterators.product(tr, tr));
x_test=MLJ.table(x_test')




X = MLJ.table(X)

y = categorical(y);


@time SVC = @load SVC pkg=LIBSVM

@time svc_mdl = SVC(kernel=k1)

svc = machine(svc_mdl, X, y)

fit!(svc);


ypred=predict(svc, x_test)


contourf(
    tr,
    tr,
    ypred,
    levels=2,
    color=cgrad(:redsblues),
    alpha=0.7
)

p1=scatter!(df[:,:x],df[:,:y],group=df[:,:color],label=false,ms=3,alpha=0.3)
#savefig(p1,"./basic4-svm.png")






















#misclassification_rate(ypred, yt)




# make data
#= n1=n2=10
Random.seed!(3203)
X = randn(20, 2)
y=vcat(fill(-1, n1), fill(1, n2))
xs,ys=X[:,1],X[:,2]
#scatter(X[:,1],X[:,2],group=y,label=false)

X = MLJ.table(X)

y = categorical(y);


@time SVC = @load SVC pkg=LIBSVM

svc_mdl = SVC(kernel=k2) #<================

#= rc = range(svc_mdl, :cost, lower=0.1, upper=5)
tm = TunedModel(model=svc_mdl, ranges=[rc], tuning=Grid(resolution=10),
                resampling=CV(nfolds=3, rng=33), measure=misclassification_rate) =#
svc = machine(svc_mdl, X, y)

fit!(svc);

ypred =predict(svc, X)

misclassification_rate(ypred, y)

 =#