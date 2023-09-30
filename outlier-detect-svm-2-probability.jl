#include("./src/memoize.jl")

import MLJ: fit!, predict, transform
import DataFrames: transform as trans
using CSV
using DataFrames
using KernelFunctions
using MLJ

using OutlierDetection
using Plots
using PrettyPrinting
using Random

urls(str) = "/Users/lunarcheung/Public/DataSets/clustering-datasets/$str.csv"
fetch(str)=urls(str)|>d->CSV.File(d,missingstring="NA")|>DataFrame|>dropmissing
df=fetch("outliers")
coerce!(df, :x=>Continuous,:y=>Continuous, :color=>Multiclass)
origin_data=select(df,[:x,:y])|>Matrix|>MLJ.table
#normal value
dfx=filter(row -> row.color in (1,0), df)
X=select(dfx,[:x,:y])|>Matrix|>MLJ.table
y=dfx[:,:color]

# outlier
o_X=filter(row -> row.color in (2), df)
outlier_X=select(o_X,[:x,:y])|>Matrix|>MLJ.table
outlier_y=o_X[:,:color]




#define model
OneClassSVM = @load OneClassSVM pkg=LIBSVM
model=OneClassSVM()
pmodel = ProbabilisticDetector(model)
#dmodel = BinaryThresholdPredictor(pmodel, threshold=0.9)
#dmach = machine(dmodel, X)|>fit!
#yhat= predict(dmach, origin_data)

#transform(mach,origin_data)
#= plt = plot();
scatter!(plt, X.x1, X.x2,group=y;label="",alpha=0.5);
scatter!(plt, outlier_X.x1, outlier_X.x2;markershape=:cross,label="outlier",); =#
#display(plt)

#schema(X)