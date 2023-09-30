"""
from   https://juliaai.github.io/DataScienceTutorials.jl/isl/lab-9/index.html
https://discourse.julialang.org/t/how-do-i-specify-a-
-kernel-in-svm-using-libsvm-or-mlj/61962
"""

include("./src/memoize.jl")

import MLJ: fit!, predict

using .Memoize
using CSV
using DataFrames
using KernelFunctions
using MLJ
using Plots
using PrettyPrinting
using Random
urls(str) = "/Users/lunarcheung/Public/DataSets/clustering-datasets/$str.csv"
f(str) = urls(str) |> CSV.File |> DataFrame
function make_svm_plot(str)
    local df = f(str)

    rows, _ = size(df)

    x1 = df[1:2:rows, :x]
    x2 = df[1:2:rows, :y]
    X = hcat(x1, x2)
    y = df[1:2:rows, :color]

    X = MLJ.table(X)
    y = categorical(y)



    #test
    n1 = n2 = 200
    xlow, xhigh = extrema(df[:, :x])
    ylow, yhigh = extrema(df[:, :y])
    tx = range(xlow, xhigh; length=n1)
    ty = range(ylow, yhigh; length=n2)
    x_test = mapreduce(collect, hcat, Iterators.product(tx, ty))
    x_test = MLJ.table(x_test')
    SVC = @load SVC pkg = LIBSVM

    svc_mdl = SVC()

    svc = machine(svc_mdl, X, y)

    fit!(svc)
    ypred = predict(svc, x_test)
    cat = df[:, :color] |> levels |> length
    contourf(tx, ty, ypred, levels=cat, color=cgrad(:redsblues), alpha=0.7)
    p1 = scatter!(df[:, :x], df[:, :y], group=df[:, :color], label=false, ms=3, alpha=0.3)
    #savefig(p1,"./images/$str-svm.png")
end

cls2 = ["basic1", "basic2", "basic3", "basic4", "basic5", "blob", "box", "boxes", "boxes3", "chrome", "dart", "dart2", "face", "ring", "sparse", "lines", "lines2", "spiral", "spiral2", "spirals", "network", "hyperplane", "supernova", "triangle", "un", "un2", "wave","isolation"]


    make_svm_plot("isolation")

