"""
from   https://juliaai.github.io/DataScienceTutorials.jl/isl/lab-9/index.html
https://discourse.julialang.org/t/how-do-i-specify-a-
-kernel-in-svm-using-libsvm-or-mlj/61962
"""

include("./src/memoize.jl")

import GLMakie: contourf!, scatter!
import MLJ: fit!, predict

using .Memoize
using CSV
using DataFrames
using GLMakie
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

#test data
n1=n2=150
xlow,xhigh=extrema(X.x1)
ylow,yhigh=extrema(X.x2)
tx = range(xlow,xhigh; length=n1)
ty = range(ylow,yhigh; length=n2)
x_test = mapreduce(collect, hcat, Iterators.product(tx, ty))
x_test=MLJ.table(x_test') 

#build model
@time  SVC = @load SVC pkg=LIBSVM
@time   svc_mdl = SVC()
@time   svc = machine(svc_mdl, X, y)
@time fit!(svc);
#ypred=predict(svc, x_test)
#ypred=Makie.convert_single_argument(ypred)
#plot
#cat=df2[:,:species]|>levels|>length

# fig = Figure()
# ax = Axis(fig[1, 1])
#co=contourf!(tx,ty,ypred,colormap=:viridis)
#scatter!(ax,X.x1,X.x2,y)
# Colorbar(fig[1, 2], co)
#savefig(p1,"./$pic-svm.png")








