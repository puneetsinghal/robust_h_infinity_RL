clear all
close all
clc

N_terms = 20  ; 

syms tspan
for i = 1 : N_terms
    
    omega = pi*i; 
    
    a(i) = sin(omega*tspan);
    b(i) = cos(omega*tspan);
    
end

control_generator = [a , b];

save control_generator.mat control_generator