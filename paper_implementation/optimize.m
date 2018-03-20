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

    epsilon  = 0.001;
    theta_previous = zeros(L,1);
    gamma = 6;

    for i = 1 : 10000

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
            break;
        end

        theta_previous = theta_current ;
    end
    filename = strcat('./theta_current2_', num2str(index), '.mat');
    save(filename, 'theta_current');
end