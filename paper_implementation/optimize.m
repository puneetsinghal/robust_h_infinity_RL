function theta_current = optimize(rho_delphi, rho_gdelphi, rho_kdelphi, rho_uphi, rho_wphi, rho_h, index)

    L = length(sigmaL([0,0,0,0])); %size of basis function
    T = length(rho_h);

    eta = zeros([T,1]);
    Z   = zeros([T,L]);

    epsilon  = 1e-7;
    theta_previous = zeros(L,1);
    
    gamma = 1;
    for i = 1 : 1000
        randomNumbers = randi([1, T], [1,1000]);
        for t = randomNumbers
            rho_i  = rho_uphi(t,:) + 0.5*theta_previous'*reshape(rho_gdelphi(t,:,:),[L,L]) + rho_wphi(t,:) - (0.5*(1/(gamma^2))*theta_previous'*reshape(rho_kdelphi(t,:,:),[L,L])) + rho_delphi(t,:); % T x L x M
            pi_i   = 0.25*theta_previous'*reshape(rho_gdelphi(t,:,:),[L,L])*theta_previous - ...
                    (0.25*(1/(gamma^2)))*(theta_previous'*reshape(rho_kdelphi(t,:,:),[L,L])*theta_previous) + ...
                    rho_h(t,:);          % T x 1 x M
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
%     filename = strcat('./theta_current2_', num2str(index), '.mat');
%     save(filename, 'theta_current');
end