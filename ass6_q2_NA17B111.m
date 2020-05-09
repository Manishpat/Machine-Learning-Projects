
clear all
clc
load('siso5unequal.mat');
nsamples=1024;
eta_y=3;
eta_u=5 ;
eta=max(eta_y,eta_u);
var_u=0.0965;
var_y=.8697;


% modifying data(scaling)

ymeas=ymeas/sqrt(var_y);
umeas=umeas/sqrt(var_u);
L=10;
ZL=[];
for i = L+1:-1:1
    ZL = [ZL ymeas(i:nsamples+i-L-1)];
end
for i = L+1:-1:1
    ZL = [ZL umeas(i:nsamples+i-L-1)];
end
[u s V] = svd(ZL/sqrt(nsamples-L),'econ');
lambda = diag(s).^2;
err_var=min(lambda);
%HYPOTHESIS----------------\

for d=size(lambda):-1:1
    lb=0;
    lbb=0;
    dof=(d+2)*(d-1)/2;
    nd=nsamples-L-(4*L+15)/6;
    for j=(2*L+3-d) :(2*L+2)
        lb=lb+lambda(j)/d;
        lbb=lbb+log(lambda(j));
    end
    t=nd*(d*log(lb)-lbb);
    if t>chi2inv(.95,dof)
        continue %rejected
    else
        d;
        break
    end
end
eta=L+1-d;
fprintf('WHEN ERROR VARIANCE OF BOTH INPUT AND OUTPUT ARE KNOWN\n')
fprintf('\nThe value of eta(order) from  Testing hypothesis  is %f\n',eta);
   
Z_eta=[];
for i = eta+1:-1:1
    Z_eta = [Z_eta ymeas(i:nsamples+i-L-1)];
end
for i = eta+1:-1:1
    Z_eta = [Z_eta umeas(i:nsamples+i-L-1)];
end
s=size(Z_eta,1);
z_eta=Z_eta/sqrt(s);


[u s v] = svd(Z_eta,'econ');
theta = v(:,end)';
theta = [theta(1:6)/sqrt(var_y) theta(7:12)/sqrt(var_u)];
mod_coeff = theta/theta(1);
fprintf('The Values of model coefficients are (lag=10)\n');
fprintf('%f\n',mod_coeff);