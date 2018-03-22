function [] = optimize(rho_delphi, rho_gdelphi, rho_kdelphi, rho_uphi, rho_wphi, rho_h, phit, index)

    % N = 25330;
    % rho_delphi = rho_delphi(1:N, :); % T x L
    % rho_gdelphi = rho_gdelphi(1:N, :, :); % T x L x L
    % rho_kdelphi = rho_kdelphi(1:N, :, :); % T x L x L
    % rho_uphi = rho_uphi(1:N, :); % T x L
    % rho_wphi = rho_wphi(1:N, :); % T x L
    % rho_h = rho_h(1:N,:); % T x 1
%%
    L = length(phit);
    T = length(rho_h);

    eta = zeros([T,1]);
    Z   = zeros([T,L]);

    epsilon  = 1e-7;
    theta_previous = zeros(L,1);
%     theta_previous = [0.3285, 1.5877, 0.2288, -0.7028, 0.4101, -1.2514,...
%         -0.5448, -0.4595, 0.4852, 0.2078, -1.3857, 1.7518,...
%         1.1000, 0.5820, 0.1950, -0.0978, -1.0295, -0.2773, -0.2169, 0.2463]';
    gamma = 1;
%     count = 0;
    for i = 1 : 1000

        for t = 1 : length(phit)
            rho_i  = rho_uphi(t,:) + 0.5*theta_previous'*reshape(rho_gdelphi(t,:,:),[L,L]) + rho_wphi(t,:) - (0.5*(1/(gamma^2))*theta_previous'*reshape(rho_kdelphi(t,:,:),[L,L])) + rho_delphi(t,:); % T x L x M
            pi_i   = 0.25*theta_previous'*reshape(rho_gdelphi(t,:,:),[L,L])*theta_previous - (0.25*(1/(gamma^2)))*(theta_previous'*reshape(rho_kdelphi(t,:,:),[L,L])*theta_previous) + rho_h(t,:);          % T x 1 x M
            Z(t,:) = rho_i;
            eta(t,:) = pi_i  ;
        end

        Ztrans =  Z' ;
        theta_current = (Ztrans*Z)\(Ztrans*eta) ;

        norm(theta_current - theta_previous)

        if(norm(theta_current - theta_previous)<epsilon)
            count = count + 1;
        else
            count = 0;
        end
        if (count == 1)
            break;
        end

        theta_previous = theta_current ;
    end
    filename = strcat('./theta_current2_', num2str(index), '.mat');
    save(filename, 'theta_current');
    save('./theta_current2', 'theta_current');
end