using Clustering, Plots, Downloads, FileIO,DataFrames,CSV,StatsPlots

url="/Users/lunarcheung/Public/DataSets/clustering-datasets/basic1.csv"

df=DataFrame(CSV.File(url))

#first(df,10)
cat=levels(df.color)

@df df scatter(:x, :y, group = :color, title = "basic", xlabel = "X", ylabel = "Y",label=false)