load('autocomp.mat');
% Z=carbdata;
data=carbdata;
max_var=0;

%applying pca to data(witout mean shifting)
[U S V]=svd(data);
%total varienca by pca
tot_var=sum(diag(S).^2)
[Fn,adj_v,cum_v]=sparsePCA(data,5,1,0,1);
%percantage of varience
var_percent=cum_v(1)/tot_var;

fprintf('Without Mean Shifting\n') 
fprintf('Cumulative Variance %f for %f non zero elements\n',cum_v,5); 
fprintf('Percentage variance of the componests are %f\n',var_percent*100);
 
 
 
 %--------------------------------------------------------------------------
 data_mean=data-mean(data);
 [U S V]=svd(data_mean);
tot_var=sum(diag(S).^2)
[Fn,adj_v,cum_v]=sparsePCA(data_mean,5,1,0,1);
var_percent1=cum_v(1)/tot_var;
fprintf('With Mean Shifting\n') 
fprintf('Cumulative Variance %f for %f non zero elements\n',cum_v,5); 
fprintf('Percentage variance of the component are%f\n',var_percent1*100); 
 