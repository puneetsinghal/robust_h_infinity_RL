function sigmaL = sigmaL(x)

x1 = x(1);
x2 = x(2);
x3 = x(3);
x4 = x(4);

t2 = x1.^2;
t3 = x2.^2;
t4 = x3.^2;
t5 = x4.^2;
sigmaL = [t2,x1.*x2,x1.*x3,x1.*x4,t3,x2.*x3,x2.*x4,t4,x3.*x4,t5,t2.*x1.*x2,t2.*x1.*x3,t2.*x1.*x4,t2.*t3,t2.*x2.*x3,t2.*x2.*x4,t2.*t4,t2.*x3.*x4,t2.*t5,t3.*x1.*x2];
