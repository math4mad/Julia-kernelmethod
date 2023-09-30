using MultivariateStats, LinearAlgebra,Symbolics

@variables   x1,x2,x3,x4

matrix=[1 0 8 4; 0 1 2 12]
X=[x1,x2,x3,x4]
eqs=matrix*X|>expand
Symbolics.solve_for([x1 + 8x3 + 4x4~42, x2 + 2x3 + 12x4~8], [x1,x2,x3,x4])

#非方阵似乎没有办法解