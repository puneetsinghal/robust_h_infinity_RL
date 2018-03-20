function sdot = dynamics(t,state)

x1 = state(1);
x2 = state(2);
x3 = state(3);
x4 = state(4);

epsilon = 0.2 ; 
global theta_current


D = (1-epsilon^2*cos(x3)^2) ; 
f = [x2 ; (-x1+epsilon*x4^2*sin(x3)) / D ; x4 ; epsilon*cos(x3)*(x1 - epsilon*x4^2*sin(x3))/D];

g = [0 ; -epsilon*cos(x3)/D ; 0 ; 1/D];

k = [0 ; 1/D ; 0 ; -epsilon*cos(x3)/D];

u   = -0.5*g'*JsigmaL(state)'*theta_current;
d   = 0.1*exp(-0.1*t)*sin(t);

sdot = f + g*u + k*d ; 

end

