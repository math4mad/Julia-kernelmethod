module  Memoize export memoize

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
end