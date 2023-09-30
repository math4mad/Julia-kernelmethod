"""
https://www.kaggle.com/datasets/dev0914sharma/customer-clustering
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
str = "/Users/lunarcheung/Public/DataSets/customer-clustering/segmentation-data.csv"
f(str) = str|> CSV.File |> DataFrame

df=f(str)|>d->first(d,10)




#= function make_svm_plot(str)
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
    savefig(p1,"./images/$str-svm.png")
end =#





