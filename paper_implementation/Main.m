clear all
close all
clc

% Generate the sample set SM
M  = 20  ;
dt = 0.033;

tspan = [0:dt:dt*600]';

T = length(tspan);

% control_generator =[sin(pi*tspan), sin(2*pi*tspan), sin(3*pi*tspan), sin(4*pi*tspan), sin(5*pi*tspan), sin(6*pi*tspan), sin(7*pi*tspan), sin(8*pi*tspan), sin(9*pi*tspan), sin(10*pi*tspan), sin(11*pi*tspan), sin(12*pi*tspan), sin(13*pi*tspan), sin(14*pi*tspan), sin(15*pi*tspan), sin(16*pi*tspan), sin(17*pi*tspan), sin(18*pi*tspan), sin(19*pi*tspan), sin(20*pi*tspan), cos(pi*tspan), cos(2*pi*tspan), cos(3*pi*tspan), cos(4*pi*tspan), cos(5*pi*tspan), cos(6*pi*tspan), cos(7*pi*tspan), cos(8*pi*tspan), cos(9*pi*tspan), cos(10*pi*tspan), cos(11*pi*tspan), cos(12*pi*tspan), cos(13*pi*tspan), cos(14*pi*tspan), cos(15*pi*tspan), cos(16*pi*tspan), cos(17*pi*tspan), cos(18*pi*tspan), cos(19*pi*tspan), cos(20*pi*tspan) exp(-pi*tspan), exp(-2*pi*tspan), exp(-3*pi*tspan), exp(-4*pi*tspan), exp(-5*pi*tspan), exp(-6*pi*tspan), exp(-7*pi*tspan), exp(-8*pi*tspan), exp(-9*pi*tspan), exp(-10*pi*tspan), exp(-11*pi*tspan), exp(-12*pi*tspan), exp(-13*pi*tspan), exp(-14*pi*tspan), exp(-15*pi*tspan), exp(-16*pi*tspan), exp(-17*pi*tspan), exp(-18*pi*tspan), exp(-19*pi*tspan), exp(-20*pi*tspan),rand([length(tspan),1])];
% control_generator =[sin(pi*tspan), sin(2*pi*tspan), sin(3*pi*tspan), sin(4*pi*tspan), sin(5*pi*tspan), sin(6*pi*tspan), sin(7*pi*tspan), sin(8*pi*tspan), sin(9*pi*tspan), sin(10*pi*tspan), sin(11*pi*tspan), sin(12*pi*tspan), sin(13*pi*tspan), sin(14*pi*tspan), sin(15*pi*tspan), sin(16*pi*tspan), sin(17*pi*tspan), sin(18*pi*tspan), sin(19*pi*tspan), sin(20*pi*tspan)];
I1 = eye(T,T);
KK = 1 : 1 : 20;

% control_generator = 0.1*[I1(:,1)*KK,-I1(:,1)*KK,rand(T,1),...
%     sin(pi*tspan), sin(2*pi*tspan), sin(3*pi*tspan), sin(4*pi*tspan), sin(5*pi*tspan), sin(6*pi*tspan), sin(7*pi*tspan), sin(8*pi*tspan), sin(9*pi*tspan), sin(10*pi*tspan), sin(11*pi*tspan), sin(12*pi*tspan), sin(13*pi*tspan), sin(14*pi*tspan), sin(15*pi*tspan), sin(16*pi*tspan), sin(17*pi*tspan), sin(18*pi*tspan), sin(19*pi*tspan), sin(20*pi*tspan),...
%     cos(pi*tspan), cos(2*pi*tspan), cos(3*pi*tspan), cos(4*pi*tspan), cos(5*pi*tspan), cos(6*pi*tspan), cos(7*pi*tspan), cos(8*pi*tspan), cos(9*pi*tspan), cos(10*pi*tspan), cos(11*pi*tspan), cos(12*pi*tspan), cos(13*pi*tspan), cos(14*pi*tspan), cos(15*pi*tspan), cos(16*pi*tspan), cos(17*pi*tspan), cos(18*pi*tspan), cos(19*pi*tspan), cos(20*pi*tspan), ...
%     exp(-pi*tspan), exp(-2*pi*tspan), exp(-3*pi*tspan), exp(-4*pi*tspan), exp(-5*pi*tspan), exp(-6*pi*tspan), exp(-7*pi*tspan), exp(-8*pi*tspan), exp(-9*pi*tspan), exp(-10*pi*tspan), exp(-11*pi*tspan), exp(-12*pi*tspan), exp(-13*pi*tspan), exp(-14*pi*tspan), exp(-15*pi*tspan), exp(-16*pi*tspan), exp(-17*pi*tspan), exp(-18*pi*tspan), exp(-19*pi*tspan), exp(-20*pi*tspan)];
control_generator = 0.1*[I1(:,1)*KK,-I1(:,1)*KK,rand(T,1), repmat(rand(T,1),1,1),...
    sin(pi*tspan), sin(2*pi*tspan), sin(3*pi*tspan), sin(4*pi*tspan), sin(5*pi*tspan), sin(6*pi*tspan), sin(7*pi*tspan), sin(8*pi*tspan), sin(9*pi*tspan), sin(10*pi*tspan), sin(11*pi*tspan), sin(12*pi*tspan), sin(13*pi*tspan), sin(14*pi*tspan), sin(15*pi*tspan), sin(16*pi*tspan), sin(17*pi*tspan), sin(18*pi*tspan), sin(19*pi*tspan), sin(20*pi*tspan),...
    cos(pi*tspan), cos(2*pi*tspan), cos(3*pi*tspan), cos(4*pi*tspan), cos(5*pi*tspan), cos(6*pi*tspan), cos(7*pi*tspan), cos(8*pi*tspan), cos(9*pi*tspan), cos(10*pi*tspan), cos(11*pi*tspan), cos(12*pi*tspan), cos(13*pi*tspan), cos(14*pi*tspan), cos(15*pi*tspan), cos(16*pi*tspan), cos(17*pi*tspan), cos(18*pi*tspan), cos(19*pi*tspan), cos(20*pi*tspan)];
%     exp(-pi*tspan), exp(-2*pi*tspan), exp(-3*pi*tspan), exp(-4*pi*tspan), exp(-5*pi*tspan), exp(-6*pi*tspan), exp(-7*pi*tspan), exp(-8*pi*tspan), exp(-9*pi*tspan), exp(-10*pi*tspan), exp(-11*pi*tspan), exp(-12*pi*tspan), exp(-13*pi*tspan), exp(-14*pi*tspan), exp(-15*pi*tspan), exp(-16*pi*tspan), exp(-17*pi*tspan), exp(-18*pi*tspan), exp(-19*pi*tspan), exp(-20*pi*tspan)];

SM = randi([1,size(control_generator,2)],[1,M]);

% numFunctions = size(control_generator,2);
numData = M*T;
U = zeros(numData,1);
W = zeros(numData,1);
X = zeros(numData,4);

inputIndex = [];
for j = 1 : floor(numData/T)
    x0     = 3*rand(4,1);
%     u      = 0.5*control_generator*rand(size(control_generator,2),1);
%     newIndex = randi(numFunctions);
%     inputIndex = [inputIndex; newIndex];
%     u      = control_generator(:, newIndex);
%     u = sin(((j))*pi*tspan);
    u      = control_generator(:,SM(j));
    w      = 0.5*rand(length(tspan),1);
    [~,x]  = ode45(@(t,x)dynamicsknown(t, x, u, w, tspan), tspan, x0);
    
%     x0 = x(end,:);
    U((j-1)*T + 1:j*T,1) = u ;
    W((j-1)*T + 1:j*T,1) = w ;
    X((j-1)*T + 1:j*T,:) = x ;
    j;
end

save systemdata.mat U W X

%%
sizeBasisFunction = length(sigmaL([0,0,0,0]));
rho_delphi = zeros(length(X)-1, sizeBasisFunction);
rho_gdelphi = zeros(length(X)-1, sizeBasisFunction, sizeBasisFunction);
rho_kdelphi = zeros(length(X)-1, sizeBasisFunction, sizeBasisFunction);
rho_uphi = zeros(length(X)-1, sizeBasisFunction);
rho_wphi = zeros(length(X)-1, sizeBasisFunction);
rho_h = zeros(length(X)-1, 1);

% X = X(1:1000,:);
disp('optimization started');
rejectedIndex = 0;
correctIndex = 0;
for t = 1 : length(X) -1
    if(rem(t,T) == 0)
%         disp(t);
        continue;
    end
%     floor(t*100/length(X))
    xt      =  X(t,:);
    xt1     =  X(t+1,:);
    phit    =  sigmaL(xt) ;
    phit1   =  sigmaL(xt1);
    Jt      =  JsigmaL(xt);
    Jt1     =  JsigmaL(xt1);
    gt      =  g(xt);
    gt1     =  g(xt1);
    kt      =  k(xt);
    kt1     =  k(xt1);
    ht      =  h(xt);
    ht1     =  h(xt1);
    ut      =  U(t);
    ut1     =  U(t+1);
    wt      =  W(t);
    wt1     =  W(t+1);

    correctIndex = correctIndex + 1;
    rho_delphi(t,:)    = phit-phit1;                                     % T x L
    rho_gdelphi(t,:,:) = ((Jt*(gt*gt')*Jt')+(Jt1*(gt1*gt1')*Jt1'))*dt/2; % T x L x L
    rho_kdelphi(t,:,:) = ((Jt*(kt*kt')*Jt')+(Jt1*(kt1*kt1')*Jt1'))*dt/2; % T x L x L
    rho_uphi(t,:)      = ((ut*gt'*Jt')+(ut1*gt1'*Jt1'))*dt/2;            % T x L
    rho_wphi(t,:)      = ((wt*kt'*Jt')+(wt1*kt1'*Jt1'))*dt/2;            % T x L
    rho_h(t,:)         = ((ht*ht)+(ht1*ht1))*dt/2;                     % T x 1
    
end
%%
save matrices_data.mat rho_delphi rho_gdelphi rho_kdelphi rho_uphi rho_wphi rho_h

%%
close all;
t = correctIndex;
theta_current = optimize(rho_delphi(1:t,:), rho_gdelphi(1:t,:,:),...
    rho_kdelphi(1:t,:,:), rho_uphi(1:t,:), rho_wphi(1:t,:), rho_h(1:t,:), t);
fprintf('Optimization done till %d steps', t);
save('./theta_current2', 'theta_current');
pause(1);
SM
testController(dt);