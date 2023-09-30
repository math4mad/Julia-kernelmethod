"""
KMML  page 52 example 1.6
"""

using Distances



# 欧几里得距离
x=[1,0]
y=[0,1]
r1=euclidean(x, y)  # =>√2

# mahalanobis距离,m距离考虑了向量之间的相关性
M=[2 1; 1 4]
r2=mahalanobis(x, y, M)  # =>2
