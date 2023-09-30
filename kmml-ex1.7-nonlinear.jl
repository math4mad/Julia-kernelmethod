"""
kmml-ex1.7  kmml-p-54
"""

import Plots:plot
using Plots,KernelFunctions

gr()
xs=[+1 ,+1, −(1), −(1), +1, 0, −(1), 0]
ys=[+1, −(1), −(1), +1, 0, −(1), 0 ,+1]
zs=[+1, +1, +1, +1, −(1),−(1), −(1), −(1)]

kernel(x)=x^2

kxs=kernel.(xs)
kys=kernel.(ys)

p1=scatter(xs, ys, mc = zs, color = :red,title="original space" )
p2=scatter(kxs, kys, mc = zs, color = :red,title="intrinsic space")

plot(p1,p2,layout=(1,2),label=true)
