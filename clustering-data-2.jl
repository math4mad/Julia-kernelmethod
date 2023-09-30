using Clustering, Plots, Downloads, FileIO,DataFrames,CSV,StatsPlots

url="/Users/lunarcheung/Public/DataSets/clustering-datasets/basic2.csv"

df=DataFrame(CSV.File(url))

#first(df,10)
#cat=levels(df.color)

@df df scatter(:x, :y, group = :color,  alpha=0.5,title = "basic2", xlabel = "X", ylabel = "Y",label=false)