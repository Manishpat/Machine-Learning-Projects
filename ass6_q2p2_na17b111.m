%NA17b111 Manish patidar
%PROCEDURE-
%1-Run IPCA and stdest code to estimate best standard deviations to scale
%ymeas and umeas
%2-scale ymeas and u meas using the std deviation we got
%3-run the DPCA(includes the hypothesis to get the correct order ) 
%4- and run the LPCA to estimates theta



clear all
clc
load('siso5unequal.mat');
D=[ymeas umeas];
stdev=std(D);
m=mean(D);
[U S V]=svd(D);
Amat=V;
D=D./sqrt(10);
n_iter=200;
nsamples=1024;
for i=1:n_iter
    stdd=stdest(Amat,D');
    D=D./(ones(1024,1)*stdd');
    [U S V]=svd(D,'econ');
    Amat=V(:,end-1:end)';
    Amat=Amat./(ones(1,1)*stdd');
    D=D.*(ones(1024,1)*stdd');
end
st_dev=stdest(Amat,D');
stdev_y=st_dev(1);
stdev_u=st_dev(2);
x0=[stdev_y stdev_u ]
%[eststd, fval] = fmincon('obj_val',x0,[],[],[],[],[],optimset('Display','iter','MaxFunEvals',50000),Amat,D)
ymeas=ymeas/stdev_y;
umeas=umeas/stdev_u;
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
theta = [theta(1:6)/stdev_y theta(7:12)/stdev_u];
mod_coeff = theta/theta(1);
fprintf('The Values of model coefficients are (lag=10)\n');
fprintf('%f\n',mod_coeff);
        