function DAqtOut = DAqt(theta,Dtheta)

theta11 = theta(1);
theta12 = theta(2);
theta13 = theta(3);
theta21 = theta(4);
theta22 = theta(5);
theta23 = theta(6);

Dtheta11 = Dtheta(1);
Dtheta12 = Dtheta(2);
Dtheta13 = Dtheta(3);
Dtheta21 = Dtheta(4);
Dtheta22 = Dtheta(5);
Dtheta23 = Dtheta(6);
%%
% theta11 = 0;
% theta12 = 0;
% theta13 = 0;
% theta21 = 0;
% theta22 = 0;
% theta23 = 0;
% 
% Dtheta11 = 0;
% Dtheta12 = 0;
% Dtheta13 = 0;
% Dtheta21 = 0;
% Dtheta22 = 0;
% Dtheta23 = 0;

%%
DAqtOut = [(-1).*Dtheta11.*cos(theta11)+(-0.5E0).*(Dtheta11+Dtheta12).*cos( ...
  theta11+theta12)+(-0.2E0).*(Dtheta11+Dtheta12+Dtheta13).*cos(theta11+ ...
  theta12+theta13),(-0.5E0).*(Dtheta11+Dtheta12).*cos(theta11+theta12)+( ...
  -0.2E0).*(Dtheta11+Dtheta12+Dtheta13).*cos(theta11+theta12+theta13),( ...
  -0.2E0).*(Dtheta11+Dtheta12+Dtheta13).*cos(theta11+theta12+theta13), ...
  Dtheta21.*cos(theta21)+0.5E0.*(Dtheta21+Dtheta22).*cos(theta21+theta22)+ ...
  0.2E0.*(Dtheta21+Dtheta22+Dtheta23).*cos(theta21+theta22+theta23), ...
  0.5E0.*(Dtheta21+Dtheta22).*cos(theta21+theta22)+0.2E0.*(Dtheta21+ ...
  Dtheta22+Dtheta23).*cos(theta21+theta22+theta23),0.2E0.*(Dtheta21+ ...
  Dtheta22+Dtheta23).*cos(theta21+theta22+theta23);(-1).*Dtheta11.*sin( ...
  theta11)+(-0.5E0).*(Dtheta11+Dtheta12).*sin(theta11+theta12)+(-0.2E0).*( ...
  Dtheta11+Dtheta12+Dtheta13).*sin(theta11+theta12+theta13),(-0.5E0).*( ...
  Dtheta11+Dtheta12).*sin(theta11+theta12)+(-0.2E0).*(Dtheta11+Dtheta12+ ...
  Dtheta13).*sin(theta11+theta12+theta13),(-0.2E0).*(Dtheta11+Dtheta12+ ...
  Dtheta13).*sin(theta11+theta12+theta13),Dtheta21.*sin(theta21)+0.5E0.*( ...
  Dtheta21+Dtheta22).*sin(theta21+theta22)+0.2E0.*(Dtheta21+Dtheta22+ ...
  Dtheta23).*sin(theta21+theta22+theta23),0.5E0.*(Dtheta21+Dtheta22).*sin( ...
  theta21+theta22)+0.2E0.*(Dtheta21+Dtheta22+Dtheta23).*sin(theta21+ ...
  theta22+theta23),0.2E0.*(Dtheta21+Dtheta22+Dtheta23).*sin(theta21+ ...
  theta22+theta23);0,0,0,0,0,0];

end