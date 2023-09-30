"""
测试一下memoize macro 的效果
"""

include("./src/memoize.jl")

import MLJ: fit!, predict

using  .Memoize
using CSV
using DataFrames

urls(str)="/Users/lunarcheung/Public/DataSets/clustering-datasets/$str.csv"
f(str)=urls(str)|>CSV.File|>DataFrame

pic="face"
@time df1= Memoize.@memoize f(pic)

function foo()
   first(df1,10)
end


@time foo()