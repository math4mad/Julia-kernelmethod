"""
https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/examples/support-vector-machine/

@momize 似乎定义了一个很好的macro, 持续研究一段时间, 看看还有什么问题
"""

using CSV
using DataFrames
using Distributions
using KernelFunctions
using LIBSVM
using LinearAlgebra
using Plots
using Random

# Set seed
Random.seed!(1234);

macro memoize(expr)
    local cache = Dict()
    local res = undef
    local params = expr.args
    #@show params
    local id = hash(params)
    if haskey(cache, id) == true
        res = cache[id]
    else
        local val = esc(expr)

        push!(cache, (id => val))
        res = cache[id]
    end

    return :($res)
end

urls(str)="/Users/lunarcheung/Public/DataSets/clustering-datasets/$str.csv"
f(str)=urls(str)|>CSV.File|>DataFrame
#@df f("blob") scatter(:x, :y, group = :color,  alpha=0.5, xlabel = "X", ylabel = "Y",label=false)
df=f("blob")

x_train = df[:, [:x, :y]]|>Matrix|>d->d[1:1000,:]
#x_test = df[:, [:x, :y]]|>Matrix|>d->d[3001:4000,:]
y_train = df[:,:color]|>vec|>d->d[1:1000]
#y_test = df[:,:color]|>vec|>d->d[3001:4000]

scatter(df[1:1000,:x], df[1:1000,:y], group = df[1:1000,:color],  alpha=0.5, xlabel = "X", ylabel = "Y",label=false)

 #k = SqExponentialKernel() ∘ ScaleTransform(1.5)

#model = svmtrain(kernelmatrix(k, x_train'), y_train; kernel=LIBSVM.Kernel.Precomputed)


 

#rn=test_range = range(floor(Int, minimum(x_train')), ceil(Int, maximum(x_train')); length=200)
#x_test = ColVecs(mapreduce(collect, hcat, Iterators.product(test_range, test_range)));

#y_pred, _ = svmpredict(model, kernelmatrix(k, x_train', x_test));

 
# plot(; lim=extrema(x_train), aspect_ratio=1)
# contourf(
#     test_range,
#     test_range,
#     y_pred;
#     levels=1,
#     color=cgrad(:redsblues),
#     alpha=0.7,
#     colorbar_title="prediction",
# )

