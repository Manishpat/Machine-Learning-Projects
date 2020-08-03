function [ atls ] = TLS( Z)
%% Hopefully TLS

N = size(Z,1);
Z_bar = mean(Z);
Zs = Z - Z_bar;
std_Z = std(Zs);
Zs = Zs./(ones(N,1)*std_Z);

covz = cov(Zs);
%that will give only eiganvector correspond to minimum eiganvalue
[atls,min_eigvalue] = eigs(covz, 1, 'smallestabs');%that will give only eiganvector correspond to minimum eiganvalue

atls = atls./(std_Z)';
atls = atls/atls(end);
atls = - atls(1:end-1,:)

