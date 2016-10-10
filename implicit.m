function [v,ds,dt,tvec,svec] = implicit(K,r,sigma,T,gamma,s_min,s_max,M,N)
% FINITE DIFFERENCE METHOD SOLVER!
% This function calculates the price of a European call option using
% Euler's method and the following stock dynamics:
%      dS(t) = r*S(t)*dt + sigma*S(t)^gamma*dW(t)
%
% Strike price, K
% Interest rate, r
% Diffusion paramter, sigma
% Final time, T
% Elasticity variable, gamma
% Min stock price, s_min
% Max stock price, s_max
% Number of stock prices (steps), M
% Number of time steps, N

ds      = s_max/M;
dt      = T/N;
tvec    = 0:dt:T;         % Length = N+1
svec    = s_min:ds:s_max; % Length = M+1

v(M+1, N+1) = 0;                        % Final value of option matrix
v(:, N+1)   = max(svec-K,0);            % Final conditions
v(M+1, :)   = s_max-K*exp(-r*(T-tvec)); % Final Boundary condition

%% Creating A and B matrix
alpha = 0.5*(sigma^2*svec.^(2*gamma))/ds^2;
beta  = (r*svec)/(2*ds);

c = -beta + alpha;
b = -2*alpha - r;
a = beta + alpha;

% Update A matrix in boundary
a(1) = 0;             % Set to 0 in order to satisfy, Final BC (important)
A    = diag(b)+diag(c(2:M+1),-1) + diag(a(1:M),1);
A    = -A*dt+eye(M+1); % Update A with time step disc

%% Calculate v-vector and update v-matrix
for t_i = N:-1:1 % Iterating backwards, since t+1 values are known
    % Create a vector B for the BC, basically zeros except last value
    B(M,1) = a(M)*dt*(s_max-K*exp(-r*(T-tvec(t_i))));
    v(1:M,t_i) = A(1:M, 1:M)\(v(1:M, t_i+1) + B);
end
