function g = g(x)

x1 = x(1);
x2 = x(2);
x3 = x(3);
x4 = x(4);

epsilon = 0.2  ;
D = (1-(epsilon^2*(cos(x3))^2)) ; 


g = [0 ; -epsilon*cos(x3)/D ; 0 ; 1/D];

end