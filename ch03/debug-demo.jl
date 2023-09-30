function bar(a,b)
  c=a+b
  d=a*b
  return c+d
end


function foo()

    println("step1")
    println("step2")
    x=bar(3,6)
    y=x*3
    println("step3:",y)
    println("step4")
    
end

foo()
