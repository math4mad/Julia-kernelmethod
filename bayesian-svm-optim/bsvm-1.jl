"""
https://turinglang.org/v0.25/tutorials/
03-bayesian-neural-network/
参照这个方法实现

这个目前来看似乎还不能优化多项式核
"""


import MLJ: fit!,predict,Continuous,Multiclass,coerce!


using CSV
using DataFrames
using Plots
using KernelFunctions
using MLJ
using PrettyPrinting
using Random
using Turing
using BenchmarkTools
using Flux
using Flux: Optimise
using Zygote



urls(str) = "/Users/lunarcheung/Public/DataSets/clustering-datasets/$str.csv"
f(str)=urls(str)|>CSV.File|>DataFrame
df=f("hyperplane")
coerce!(df, :x=>Continuous,:y=>Continuous, :color=>Multiclass)
# train data
train_df=df[1:2:1700,:]
X=select(train_df,[:x,:y])|>Matrix|>MLJ.table
y=df[1:2:1700,3]

# test data
test_df=df[1:3:1000,:] 
test_X=select(test_df,[:x,:y])|>Matrix|>MLJ.table
test_y=df[1:3:1000,:color]


#plot   origin data
n1=n2=100
xlow,xhigh=extrema(X.x1)
ylow,yhigh=extrema(X.x2)
tx = range(xlow,xhigh; length=n1)
ty = range(ylow,yhigh; length=n2)
x_test = mapreduce(collect, hcat, Iterators.product(tx, ty))
x_test=MLJ.table(x_test') 


function kernel_creator(θ)
    return PolynomialKernel(; degree=θ[1], c=θ[2])
end

# function kernel_creator(θ)
#     return (exp(θ[1]) * SqExponentialKernel() + exp(θ[2]) * Matern32Kernel()) ∘
#            ScaleTransform(exp(θ[3]))
# end


function fx(x_test, x_train, y_train, θ)
    k = kernel_creator([4,1])
    SVC = @load SVC pkg=LIBSVM
    svc_mdl = SVC(;kernel=k)
    svc = machine(svc_mdl, x_train, y_train)
    fit!(svc);
    ypred=predict(svc, x_test)
    return ypred
end

θ=[1.1, 0.1, 0.01, 0.001]


function loss(θ)
    ypred=fx(X, X, y, θ)
    return misclassification_rate(ypred, y)
end


ypred=fx(x_test,X,y,θ)


contourf(
    tx,
    ty,
    ypred,
    levels=1,
    color=cgrad(:redsblues),
    alpha=0.7
)
p1=scatter!(df[:,:x],df[:,:y],group=df[:,:color],label=false,ms=3,alpha=0.3)









