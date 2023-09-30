"""
为什么要在微积分中使用实数域定义?
微积分的核心是要研究函数的行为, 如果函数变化太快, 研究就不太容易实现
对于悬垂的沥青, 滴落的液滴变化时间尺度可能以数十年来计算
对于蜗牛, 运动时间变化以秒或者分钟计算
对于光, 波运动时间以皮秒来计算
实数系的无限可分为研究宇宙中所有的变化提供尺度, 不管变化率有多大, 我们总可以让Δx->0,无限小, 从而得到因变量的微小变化
在进行微分计算的时候, 误差就会降低到最低程度. 

本例近似曲线下面积
"""

import GLMakie:wireframe!
using GLMakie,Distributions
w=0.2
xs=Vector(range(0,5,100))
xs2=Vector(1:w:4)
f(x)=ℯ^x
ys=f.(xs)

"""
## define Rect
rect = Rect(start-x,start-y,w,height)
小矩形的初始点的 y坐标始终为 0
"""
function  rect(;x=0,w=w)
    return  Rect(x,0,w,f(x))
end
boxs=[rect(;x=x,w=w) for x in  xs2]

function plot_res()
    fig=Figure()
    ax=Axis(fig[1,1])
    lines!(ax,xs,ys)
    for  i in eachindex(boxs)
        mesh!(ax, boxs[i], color=(:lightblue,0.2))
        wireframe!(ax, boxs[i]; color = :black, transparency=true)
    end
    fig
    #save("exp-function-method.png",fig)
end

plot_res()



