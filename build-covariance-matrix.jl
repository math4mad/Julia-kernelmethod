"""
Surrogates-Gaussian Process Modeling, Design, and Optimization for the Applied SciencesCRC (2020).pdf
page 161
"""

import StatsPlots: plot
using FillArrays
using Distributions
using GLMakie
using KernelFunctions

k = SqExponentialKernel()
n=100
x = Vector(1:n)
Σ=kernelmatrix(k, x)

mu=Fill(3,n)
d=MvNormal(mu,Σ) 

y=rand(d,1)

plot(x,y)