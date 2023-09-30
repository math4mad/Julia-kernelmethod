"""
bsvm-with-turing.jl 中 svm在最后的模型中,目录数组无法转换, 
这里试试用 logistic 的方法
"""

macro memoize(expr)
    local cache = Dict()
    local res = undef
    local params = expr.args
    #@show params
    local id = hash(params)
    if haskey(cache, id) == true
        res = cache[id]
    else
        local val = esc(expr)

        push!(cache, (id => val))
        res = cache[id]
    end

    return :($res)
end

using MLJ
using CSV
using DataFrames
using Distributions
using Random
using Turing   
using MCMCChains, Plots, StatsPlots 
using StatsFuns: logistic

Random.seed!(0);


urls(str) = "/Users/lunarcheung/Public/DataSets/clustering-datasets/$str.csv"
get(str)=urls(str)|>CSV.File|>DataFrame
df=get("un")
#coerce!(df, :color=>Multiclass)
# train data
train_df=df[1:2:2000,:]
X=select(train_df,[:x,:y])|>Matrix|>MLJ.table

y=train_df[:,3]

# test data
test_df=df[1:3:1000,:] 
test_X=select(test_df,[:x,:y])|>Matrix|>MLJ.table
test_y=test_df[:,:color]


#plot   origin data
# n1=n2=100
# xlow,xhigh=extrema(X.x1)
# ylow,yhigh=extrema(X.x2)
# tx = range(xlow,xhigh; length=n1)
# ty = range(ylow,yhigh; length=n2)
# x_test = mapreduce(collect, hcat, Iterators.product(tx, ty))
# x_test=MLJ.table(x_test') 



rows,cols=size(train_df)

@model function logistic_regression(x, y, n=rows, σ=0.2)
    intercept ~ Normal(0, σ)

    a ~ Normal(0, σ)
    b~ Normal(0, σ)
    

    for i in 1:n
        v = logistic(intercept + a * sqrt(x.x1[i]) + b* sqrt(x.x2[i]))
        y[i] ~ Bernoulli(v)
    end
end;


m = logistic_regression(X, y, rows, 1)

chain =sample(m, NUTS(), MCMCThreads(), 1_500, 3)

#plot(chain)

function predictions(x, chain, threshold)
    # Pull the means from each parameter's sampled values in the chain.
    intercept = mean(chain[:intercept])
    a = mean(chain[:a])
    b = mean(chain[:b])
    
    rows=334

    v = []


    for i in 1:334
        num = logistic(
            intercept + a * sqrt(x.x1[i]) + b* sqrt(x.x2[i])
        )
        if num >= threshold
            push!(v,1)
        else
            push!(v,0)
        end
    end
    return v
end;

threshold = 0.07

#pred = predictions(test_X, chain, threshold)

#lossf = sum((pred - test_y) .^ 2) / length(test_y)

#chain(:a,:b)

df=DataFrame(chain)


