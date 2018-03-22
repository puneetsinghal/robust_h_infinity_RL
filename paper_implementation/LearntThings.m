% clear all
% close all
% clc

global w tspan
tspan  = 0:0.033:100 ;
x0     = rand(4,1); 
% x0 = randi([-3,3], [4,1]);
load theta_current2.mat 

global theta_current
[t,x]  = ode23s(@dynamics,tspan,x0);

figure(1)
plot(t,x(:,1),'-b')
hold on
plot(t,x(:,2),'-r')
plot(t,x(:,3),'-m')
plot(t,x(:,4),'-k')
legend('x1','x2','x3','x4');
grid on
xlabel('Time [s]');
ylabel('States');
title('Off Policy RL');

