"""
https://juliagaussianprocesses.github.io/KernelFunctions.jl
/stable/examples/train-kernel-parameters/
"""

import Plots:plot!
using KernelFunctions
using LinearAlgebra
using Distributions
using Plots
using BenchmarkTools
using Flux
using Flux: Optimise
using Zygote
using Random: seed!
seed!(42);

xmin, xmax = -3, 3  # Bounds of the data
N = 50 # Number of samples
x_train = rand(Uniform(xmin, xmax), N)  # sample the inputs
σ = 0.1
y_train = sinc.(x_train) + randn(N) * σ  # evaluate a function and add some noise
x_test = range(xmin - 0.1, xmax + 0.1; length=300)


function plot_raw_data()
    scatter(x_train, y_train; label="data")
    plot!(x_test, sinc; label="true function")
end

#plot_raw_data()


function kernel_creator(θ)
    return (exp(θ[1]) * SqExponentialKernel() + exp(θ[2]) * Matern32Kernel()) ∘
           ScaleTransform(exp(θ[3]))
end

function f(x, x_train, y_train, θ)
    k = kernel_creator(θ[1:3])
    return kernelmatrix(k, x, x_train) *
           ((kernelmatrix(k, x_train) + exp(θ[4]) * I) \ y_train)
end

function plot_yhat()
    p0 = [1.1, 0.1, 0.01, 0.001]
    θ = log.(p0)
    ŷ = f(x_test, x_train, y_train, θ)
    plot_raw_data()
    plot!(x_test, ŷ; label="prediction")

end

#plot_yhat()

function loss(θ)
    ŷ = f(x_train, x_train, y_train, θ)
    return norm(y_train - ŷ) + exp(θ[4]) * norm(ŷ)
end





function running_time()
    @benchmark let
        θ = log.(p0)
        opt = Optimise.ADAGrad(0.5)
        grads = only((Zygote.gradient(loss, θ)))
        Optimise.update!(opt, θ, grads)
    end
end
#running_time()

p0 = [1.1, 0.1, 0.01, 0.001]
θ = log.(p0) # Initial vector
opt = Optimise.ADAGrad(0.5)

anim = Animation()
for i in 1:15
    grads = only((Zygote.gradient(loss, θ)))
    Optimise.update!(opt, θ, grads)
    scatter(
        x_train, y_train; lab="data", title="i = $(i), Loss = $(round(loss(θ), digits = 4))"
    )
    plot!(x_test, sinc; lab="true function")
    plot!(x_test, f(x_test, x_train, y_train, θ); lab="Prediction", lw=3.0)
    frame(anim)
end
gif(anim, "train-kernel-param.gif"; show_msg=false, fps=15);





