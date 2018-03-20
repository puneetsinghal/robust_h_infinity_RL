function xdot = dynamicsknown(t,x)

global w u tspan

ut = interp1(tspan,u,t);
wt = interp1(tspan,w,t);

epsilon = 0.2  ;
x1      = x(1) ;
x2      = x(2) ;
x3      = x(3) ;
x4      = x(4) ;

D = (1-(epsilon^2*(cos(x3))^2)) ; 
f = [x2 ; (-x1+epsilon*x4^2*sin(x3)) / D ; x4 ; epsilon*cos(x3)*(x1 - epsilon*x4^2*sin(x3))/D];
g = [0 ; -epsilon*cos(x3)/D ; 0 ; 1/D];
k = [0 ; 1/D ; 0 ; -epsilon*cos(x3)/D];

% ut   = -0.5*g'*JsigmaL(x)'*theta_current;

xdot = f + g*ut + k*wt;

end

