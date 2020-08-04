function [sd] = stdest(A,Y);
%  function for generating initial estimate of error covariance matrix
%  based on Keller's method (least squares solution)
% INPUTS:
% A : m x n constraint matrix
% Y : data matrix n x N, n rows are variables and N columns are samples
% %
% OUTPUTS
%  sd : Estimated measurement error standard deviations
%
[m n] = size(A);
nz = n;
maxnz = m*(m+1)/2;
if ( nz > maxnz )
        disp ('The maximum number of nonzero elements of Qe that can be estimated exceeds limit');
        return
end
% Construct D matrix
G = [];
for j = 1:n
    C = [];
    for i = 1:m
        C = [C; A(i,j)*A(:,j)];
    end
    G = [G, C];
end
% Construct RHS of equation
nsamples = size(Y,2);
R = A*Y;
V = R*R'/nsamples;
vecV = [];
for i = 1:m
    vecV = [vecV; V(:,i)];
end
% Least squares estimate
sd = sqrt(abs((G'*G)\(G'*vecV)));

