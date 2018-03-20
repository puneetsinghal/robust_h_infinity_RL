% clear all
% close all
% clc

setA = 0:1:4;
setB = 0:1:4;
setC = 0:1:4;
setD = 0:1:4;


combination = [];
for a = 1 : length(setA)
    for b = 1 : length(setB)
        for c = 1 : length(setC)
            for d = 1 : length(setD)
                
                term = setA(a) + setB(b) + setC(c) + setD(d) ;
                if(term~=0 && term <=4 && rem(term,2)==0)
                    combination = [setA(a) setB(b) setC(c) setD(d);combination];
                end
            end
        end
    end
end



syms x1 x2 x3 x4 'real'

for i = 1 : size(combination,1)   
sigmaL(i) = (x1^(combination(i,1)))*(x2^(combination(i,2)))*(x3^(combination(i,3)))*(x4^(combination(i,4)));
end

f =  matlabFunction(sigmaL,'File','sigmaL');



JsigmaL = jacobian(sigmaL,[x1,x2,x3,x4]);

f =  matlabFunction(JsigmaL,'File','JsigmaL');