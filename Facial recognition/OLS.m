function [aols] = OLS(X,Y)

N = size(X,1);
Xs = X - mean(X);% shifting the data with mean of data
Xs = Xs./(ones(N,1)*std(X));
Y_bar = mean(Y);
Ys = Y-Y_bar; 
Xs = Ys./(ones(N,1)*std(Y));%NO NEED TO SCALING TEMPRATURE
%x = round(per*m); 

% A1 = [dat_1(:,1:(n-1)) ,ones(m,1)];
% A2 = [data_1(:,1:(n-1)) ,ones(m,1)];
aols = ((Xs')*Xs)\(Xs')*Ys;
aols = aols./(std(X))'



end

