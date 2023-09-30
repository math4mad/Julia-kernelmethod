"""
加载 `kaggle` 下载的分类和聚类生成数据
"""

using Clustering, Plots, Downloads, FileIO,DataFrames,CSV,StatsPlots

cls=["basic1","basic2","basic3","basic4","basic5","blob"]
cls2=["box","boxes","boxes3","chrome","dart","dart2","face","ring","sparse","lines","lines2"]
cls3=["spiral","spiral2","spirals","network","hyperplane","supernova","triangle","un","un2","wave","outlier"]
cls=["basic1","basic2","basic3","basic4","basic5","blob","box","boxes","boxes3","chrome","dart","dart2","face","ring","sparse","lines","lines2","spiral","spiral2","spirals","network","hyperplane","supernova","triangle","un","un2","wave","outlier"]

urls(str)="/Users/lunarcheung/Public/DataSets/clustering-datasets/$str.csv"
f(str)=urls(str)|>CSV.File|>DataFrame
#|>d->first(d,10)

@df f("dart2") scatter(:x, :y, group = :color,  alpha=0.5, xlabel = "X", ylabel = "Y",label=false)

