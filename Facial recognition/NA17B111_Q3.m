clc
clear
close
%% 
S = [7,21,34;21,64,102;34,102,186];
[V,D] = eig(S);
D = diag(D)
%% 2nd question 
(D(3))/sum(D) > 0.95 
%% Constraint equations
[U,Z,V] = svd(S);
U(:,2:3)'
V
Z
U(:,1)
%% Scores
Score= ([10.1 , 73, 135.5]-[9,68,129])*U(:,1)
%%  Solving for mass , length using two equations
A = [U(:,2:3)',-U(:,2:3)'*[9;68;129]];
Ans = -[A(:,1),A(:,2)]\((A(:,3)*73)+A(:,end))
%%
B = U(:,3);
mass = -[73,135.5]*B(2:3)/(B(1));