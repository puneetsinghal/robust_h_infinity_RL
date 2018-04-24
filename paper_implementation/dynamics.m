function sdot = dynamics(t, x, weights)

x1 = x(1);
x2 = x(2);
x3 = x(3);
x4 = x(4);

epsilon = 0.2 ; 

D = (1-epsilon^2*cos(x3)^2) ; 
f = [x2 ; (-x1+epsilon*x4^2*sin(x3)) / D ; x4 ; epsilon*cos(x3)*(x1 - epsilon*x4^2*sin(x3))/D];

g = [0 ; -epsilon*cos(x3)/D ; 0 ; 1/D];

k = [0 ; 1/D ; 0 ; -epsilon*cos(x3)/D];

u   = -0.5*g'*JsigmaL(x)'*weights;
d   = 0.0*exp(-0.1*t)*sin(t);

sdot = f + g*u + k*d ; 

end

