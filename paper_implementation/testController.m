function [] = testController(dt)
    
    if (nargin ==0)
        dt = 0.033;
    end
    tspan  = 0:dt:1000;
    x0     = rand(4,1);
    weights = load('theta_current2.mat');
    weights = weights.theta_current;
    
    [t,x]  = ode45(@(t, x)dynamics(t, x, weights), tspan, x0);

    figure();
    hold on
    
    plot(t,x(:,1),'-b');
    plot(t,x(:,2),'-r');
    plot(t,x(:,3),'-m');
    plot(t,x(:,4),'-k');
    legend('x1','x2','x3','x4');
    grid on
    xlabel('Time [s]');
    ylabel('States');
    title('Off Policy RL');
end