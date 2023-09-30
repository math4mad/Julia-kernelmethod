"""
https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/examples/support-vector-machine/
with marco
"""

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


n1 = n2 = 50;

angle1 = range(0, π; length=n1)
angle2 = range(0, π; length=n2)
X1 = [cos.(angle1) sin.(angle1)] .+ 0.1 .* randn.()
X2 = [1 .- cos.(angle2) 1 .- sin.(angle2) .- 0.5] .+ 0.1 .* randn.()
X = [X1; X2]
x_train =  RowVecs(X)
y_train =  vcat(fill(-1, n1), fill(1, n2));


## define kernel method
k =  SqExponentialKernel() ∘ ScaleTransform(1.5)


model=  svmtrain(kernelmatrix(k, x_train), y_train; kernel=LIBSVM.Kernel.Precomputed)

 test_range = range(floor(Int, minimum(X)), ceil(Int, maximum(X)); length=100)
 x_test =  ColVecs(mapreduce(collect, hcat, Iterators.product(test_range, test_range)));

@time y_pred, _ = svmpredict(model, kernelmatrix(k, x_train, x_test));


plot(; lim=extrema(test_range), aspect_ratio=1)
contourf!(
    test_range,
    test_range,
    y_pred;
    levels=1,
    color=cgrad(:redsblues),
    alpha=0.7,
    colorbar_title="prediction",
)
scatter!(X1[:, 1], X1[:, 2]; color=:red, label="training data: class –1")
scatter!(X2[:, 1], X2[:, 2]; color=:blue, label="training data: class 1")


