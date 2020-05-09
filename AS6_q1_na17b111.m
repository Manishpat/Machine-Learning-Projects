


clear all
clc
load('siso5equal.mat');
nsamp=1024;
eta_y=3;
eta_u=5 ;
eta=max(eta_y,eta_u);


%-------------------------------------------
%PART_A
Z_y = [ymeas(eta+1:nsamp) ymeas(eta-2:nsamp-3) umeas(eta-2:nsamp-3) umeas(eta-4:nsamp-5)];
s=size(Z_y,1);
Z=[Z_y]/sqrt(s);
  size(Z);
  
[U S V] = svd(Z,'econ');
eig_val = diag(S).^2;
theta = V(:,end)';
mod_coeff = theta/theta(1);
fprintf('The Value of model coefficients for PART A is \n');
fprintf('%f\n',mod_coeff);



 








%---------------------
%PART_B
%getting a proper lagged data matrix(we know eta AS 5)

Z_y = [ymeas(eta+1:nsamp) ymeas(eta:nsamp-1) ymeas(eta-1:nsamp-2) ymeas(eta-2:nsamp-3) ymeas(eta-3:nsamp-4) ymeas(eta-4:nsamp-5)];
Z_u  =[umeas(eta+1:nsamp) umeas(eta:nsamp-1) umeas(eta-1:nsamp-2) umeas(eta-2:nsamp-3) umeas(eta-3:nsamp-4) umeas(eta-4:nsamp-5)];
s=size(Z_y,1);
Z=[Z_y Z_u]/sqrt(s);
size(Z);

%applying svd on laged data matrix
[U S V] = svd(Z,'econ');
eig_val = diag(S).^2;%values of lambdas
err_var=min(eig_val);% error variance
theta = V(:,end)';
%COEFFICIENTS of model are
mod_coeff = theta/theta(1);
fprintf('The Value of model coefficients by LPCA for PART B is \n');
fprintf('%f\n',mod_coeff);


% perform a bootstrap (using 100 bootstrap sets) 
stats = bootstrp(100,@(x)[lpca(x)],Z);
%mean and stds of bootstrpaing matri to confidence interval.
mean1=mean(stats);
stds1= std(stats);
%hypothesis using  95 p confidence interval
c=0;
j=[];
for i=1:12
    means=mean(stats(:,i));
    stds=std(stats(:,i));
    a=means+2.16*stds;
    b=means-2.16*stds;
    if c>min(a,b)& c<max(a,b)
       p=1;%rejected
    else
        j=[j i]; %postions where its does not rejected 
        
    end
end
fprintf('significants ordrs are\n');
fprintf('%i \n',j);

fprintf('The Value of model coefficients for PART B after performing a bootstrap and finding the significant positions is \n');
fprintf('%f\n',mean1(j));
mean1(j);
fprintf('estimated value of error variance is %f\n\n',err_var);

%--------------------------------------------------------------------

fprintf('PART C\n');
fprintf('for lag=10\n');

L=10;%lag igven
Z_Lag=[];
for i = L+1:-1:1
    Z_Lag = [Z_Lag ymeas(i:nsamp+i-L-1)];
end
for i = L+1:-1:1
    Z_Lag = [Z_Lag umeas(i:nsamp+i-L-1)];
end
s=size(Z_Lag,1);
z_Lag=Z_Lag/sqrt(s);
[u s V] = svd(Z_Lag,'econ');
eig_val = diag(s).^2;%values of lambdas

%HYPOTHESIS----------------------------------

for d=size(eig_val):-1:1
    lb=0;
    lbb=0;
    dof=(d+2)*(d-1)/2;
    nd=nsamp-L-(4*L+15)/6;
    for j=(2*L+3-d) :(2*L+2)
        lb=lb+eig_val(j)/d;
        lbb=lbb+log(eig_val(j));
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
fprintf('The value of eta(order) from hypothesis  is %f\n',eta);
    
Zeta=[];
for i = eta+1:-1:1
    Zeta = [Zeta ymeas(i:nsamp+i-L-1)];
end
for i = eta+1:-1:1
    Zeta = [Zeta umeas(i:nsamp+i-L-1)];
end
s=size(Zeta,1);
zeta=Zeta/sqrt(s);
%performing svd
[u s v] = svd(Zeta,'econ');
theta = v(:,end)';
eig_val = diag(s).^2;
err_var=min(eig_val);%error varaience
fprintf('error variance %f\n',err_var);
mod_coeff= theta/theta(1);
fprintf('The Value of model coefficients by LPCA for PART C is (lag=10)\n');
fprintf('%f\n',mod_coeff);


%----------------------------------------------
fprintf('\n\nfor lag=15\n');
%for lag of 15
L=15;
Z_Lag=[];
for i = L+1:-1:1
    Z_Lag = [Z_Lag ymeas(i:nsamp+i-L-1)];
end
for i = L+1:-1:1
    Z_Lag = [Z_Lag umeas(i:nsamp+i-L-1)];
end
[u s V] = svd(Z_Lag/sqrt(nsamp-L),'econ');
eig_val = diag(s).^2;
%HYPOTHESIS-------------------------------------------

for d=size(eig_val):-1:1
    lb=0;
    lbb=0;
    dof=(d+2)*(d-1)/2;
    nd=nsamp-L-(4*L+15)/6;
    for j=(2*L+3-d) :(2*L+2)
        lb=lb+eig_val(j)/d;
        lbb=lbb+log(eig_val(j));
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
fprintf('The value of eta(order) from hypothesis  is %f\n',eta);

    
Zeta=[];
for i = eta+1:-1:1
    Zeta = [Zeta ymeas(i:nsamp+i-L-1)];
end
for i = eta+1:-1:1
    Zeta = [Zeta umeas(i:nsamp+i-L-1)];
end
[u s v] = svd(Zeta/sqrt(nsamp-eta),'econ');
theta = v(:,end)';
eig_val = diag(s).^2;
err_var=min(eig_val);%error varaience
fprintf('error variance %f\n',err_var);
mod_coeff = theta/theta(1);
fprintf('The Value of model coefficients by LPCA for PART C is (lag=15)\n');
fprintf('%f\n',mod_coeff);

  