function DB1qOut = DB1q(theta)

theta11 = theta(1);
theta12 = theta(2);
theta13 = theta(3);
theta21 = theta(4);
theta22 = theta(5);
theta23 = theta(6);
%%
% theta11 = 0;
% theta12 = 0;
% theta13 = 0;
% theta21 = 0;
% theta22 = 0;
% theta23 = 0;
% 
%%
DB1qOut = [(0.2E1.*cos(theta11+(-1).*theta21)+0.1E1.*cos(theta11+(-1).*theta21+( ...
  -1).*theta22)+0.4E0.*cos(theta11+(-1).*theta21+(-1).*theta22+(-1).* ...
  theta23)).*csc(theta12),csc(theta12).*(0.4E0.*cos(theta12+theta13)+cot( ...
  theta12).*((-0.4E0).*sin(theta12+theta13)+(-0.2E1).*sin(theta11+(-1).* ...
  theta21)+(-0.1E1).*sin(theta11+(-1).*theta21+(-1).*theta22)+(-0.4E0).* ...
  sin(theta11+(-1).*theta21+(-1).*theta22+(-1).*theta23))),0.4E0.*cos( ...
  theta12+theta13).*csc(theta12);(0.1E1.*cos(theta11+(-1).*theta21+(-1).* ...
  theta22)+0.4E0.*cos(theta11+(-1).*theta21+(-1).*theta22+(-1).*theta23)) ...
  .*csc(theta12),csc(theta12).*(0.4E0.*cos(theta12+theta13)+cot(theta12).* ...
  ((-0.4E0).*sin(theta12+theta13)+(-0.1E1).*sin(theta11+(-1).*theta21+(-1) ...
  .*theta22)+(-0.4E0).*sin(theta11+(-1).*theta21+(-1).*theta22+(-1).* ...
  theta23))),0.4E0.*cos(theta12+theta13).*csc(theta12);0.4E0.*cos(theta11+ ...
  (-1).*theta21+(-1).*theta22+(-1).*theta23).*csc(theta12),csc(theta12).*( ...
  0.4E0.*cos(theta12+theta13)+cot(theta12).*((-0.4E0).*sin(theta12+ ...
  theta13)+(-0.4E0).*sin(theta11+(-1).*theta21+(-1).*theta22+(-1).* ...
  theta23))),0.4E0.*cos(theta12+theta13).*csc(theta12)];

end

