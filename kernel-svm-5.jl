"""
from   https://juliaai.github.io/DataScienceTutorials.jl/isl/lab-9/index.html
https://discourse.julialang.org/t/how-do-i-specify-a-
-kernel-in-svm-using-libsvm-or-mlj/61962
"""

import MLJ: fit!,predict

using MLJ
using Plots
using PrettyPrinting
using Random
using KernelFunctions

#define kernelmethods
k1=PolynomialKernel(; degree=2, c=1)
k2 = SqExponentialKernel() ∘ ScaleTransform(1.5)
# make data
n1=n2=10
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

