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
df=f("basic5")

X = df[1:2:2000, [:x, :y]]|>Matrix
x_train=RowVecs(X)
#x_test = df[:, [:x, :y]]|>Matrix|>d->d[3001:4000,:]
y_train = df[1:2:2000,:color]|>vec
#y_test = df[:,:color]|>vec|>d->d[3001:4000]
#scatter(df[1:500,:x], df[1:500,:y], group = df[1:500,:color],  alpha=0.5, xlabel = "X", ylabel = "Y",label=false)

k = SqExponentialKernel() ∘ ScaleTransform(1.5)
#k=PolynomialKernel(; degree=2, c=1)
model = svmtrain(kernelmatrix(k, x_train), y_train; kernel=LIBSVM.Kernel.Precomputed)


 

#rn=test_range = range(floor(Int, minimum(x_train')), ceil(Int, maximum(x_train')); length=1000)
t =  df[2001:3000,[:x, :y]]|>Matrix
x_test=RowVecs(t)
y_pred, _ = svmpredict(model, kernelmatrix(k, x_train, x_test));
# # yt=Array(y_pred)



contourf(
    t[:,1],
    t[:,2],
    y_pred,
    levels=1,
    
    color=cgrad([:orange, :blue], [0.1, 0.3,0.5, 0.8]),
    alpha=0.7,
    colorbar_title="prediction",
)
#scatter!(df[1:500,:x], df[1:500,:y], group = df[1:500,:color],  alpha=0.5, xlabel = "X", ylabel = "Y",label=false)



