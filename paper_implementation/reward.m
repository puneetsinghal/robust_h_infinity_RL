function H = reward(x,u,d,gamma)

H = x'*x + u'*u - gamma^2*d^2 ; 

end

