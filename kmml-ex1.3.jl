using LinearAlgebra

X=[1 0;
   2 0;
   0 -1
]
y=vec([1 ,-1])


#w̅=(inv(X'*X)*X')'y

#wX'*inv(X*X')*X'*y

#w=(inv(X'*X)*X')'*y

inv(X'*X)*X'*y 