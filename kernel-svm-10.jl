"""
Palm Penguins
"""

include("./src/memoize.jl")


import MLJ: fit!, predict

using .Memoize
using CSV
using DataFrames
using Plots
using KernelFunctions
using MLJ
using PrettyPrinting
using Random





str="/Users/lunarcheung/Public/DataSets/palmerpenguins.csv"
str2="/Users/lunarcheung/Public/DataSets/penguins.csv"

fetch(str)=str|>d->CSV.File(d,missingstring="NA")|>DataFrame|>dropmissing

df2=fetch(str2)|>d->coerce(d, :bill_length_mm => Continuous, :bill_depth_mm => Continuous,:species=>Multiclass)
X=df2[:,[:bill_length_mm,:bill_depth_mm]]|>Matrix|>MLJ.table
cat=unique(df2[:,:species])
y=df2[:,:species]




#test
n1=n2=150
xlow,xhigh=extrema(X.x1)
ylow,yhigh=extrema(X.x2)
tx = range(xlow,xhigh; length=n1)
ty = range(ylow,yhigh; length=n2)
x_test = mapreduce(collect, hcat, Iterators.product(tx, ty))
x_test=MLJ.table(x_test') 


SVC = @load SVC pkg=LIBSVM
svc_mdl = SVC()
svc = machine(svc_mdl, X, y;)
fit!(svc);


ypred = predict(svc, x_test)
cat = df2[:, :species] |> levels |> length
contourf(tx, ty, ypred, levels=cat, color=cgrad(:redsblues), alpha=0.7)
p1 = scatter!(df2[:, :bill_length_mm], df2[:, :bill_depth_mm], group=df2[:, :species], label=false, ms=3, alpha=0.3)

#savefig(p1,"./images/palmre-penguin-svm.png")








