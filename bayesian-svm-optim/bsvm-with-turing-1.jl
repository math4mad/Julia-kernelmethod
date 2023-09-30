"""
https://turinglang.org/v0.25/tutorials/
03-bayesian-neural-network/
参照这个方法实现
"""


import MLJ:  Multiclass, coerce!, fit!, predict

using BenchmarkTools
using CSV
using DataFrames
using Distributions
using KernelFunctions
using MLJ
using Plots
using PrettyPrinting
using Random
using Turing    




urls(str) = "/Users/lunarcheung/Public/DataSets/clustering-datasets/$str.csv"
get(str)=urls(str)|>CSV.File|>DataFrame
df=get("hyperplane")
coerce!(df, :color=>Multiclass)
# train data
train_df=df[1:2:1000,:]
X=select(train_df,[:x,:y])|>Matrix|>MLJ.table
y=train_df[:,3]

# test data
test_df=df[1:3:1000,:] 
test_X=select(test_df,[:x,:y])|>Matrix|>MLJ.table
test_y=test_df[:,:color]


#plot   origin data
# n1=n2=100
# xlow,xhigh=extrema(X.x1)
# ylow,yhigh=extrema(X.x2)
# tx = range(xlow,xhigh; length=n1)
# ty = range(ylow,yhigh; length=n2)
# x_test = mapreduce(collect, hcat, Iterators.product(tx, ty))
# x_test=MLJ.table(x_test') 


function kernel_creator(θ)
    return PolynomialKernel(; degree=θ[1], c=θ[2])
end

function f(x_train,y_train,θ::Array)
        k = kernel_creator(θ[1:2])
        SVC = @load SVC pkg=LIBSVM
        svc_mdl = SVC(kernel=k)
        svc = machine(svc_mdl, x_train, y_train,scitype_check_level=0)
        fit!(svc);
        
        return  x_test->predict(svc, x_test)
    
end

@model function bayes_svm(x_train,y_train,x_test,y_test,f)
       X=x_train
       y=y_train

       degree~DiscreteUniform(1,10)
       c~DiscreteUniform(0,4)
       pred_func=f(X,y,[degree,c])

       yhat=pred_func(x_test)
       @show yhat
       for i in eachindex(y_test)
        y_test[i]~Categorical(yhat[i])
       end
       

end

N=2000

sample(
    bayes_svm(X,y,test_X,test_y,f),NUTS(), N
)








# function plot_img()
#     contourf(
#         tx,
#         ty,
#         ypred,
#         levels=1,
#         color=cgrad(:redsblues),
#         alpha=0.7
#     )
#     p1=scatter!(df[:,:x],df[:,:y],group=df[:,:color],label=false,ms=3,alpha=0.3)
    
# end











