function [v,err] = mc(T,S0,K,r,sigma,gamma,M,N,Z)
% Monte Carlo solver!
% This function calculates the price of a European call option using
% Euler's method and the following stock dynamics:
%      dS(t) = r*S(t)*dt + sigma*S(t)^gamma*dW(t)
%
%T     - Time to maturity
%M     - Number of time teps
%N     - Number of paths
%S0    - Initial stock price
%K     - Strike price
%r     - Risk-free interest rate
%sigma - Volatility
%gamma - Elasticity

dt = T/M;               % Time steps
t = 0:dt:T;             % Time vector

S(N,1) = 0;
S(:,1) = S0;

% Stepping of Monte-Carlo, updating all sample paths simultaniously
for i = 1:M
    dw = Z(:,i)*sqrt(dt);    
    S = S + r*S*dt + sigma*(S.^gamma).*dw; 
end

VT = max(S-K,0);
E = mean(VT);
v = exp(-r*T)*E;

V0true = bsexact(sigma,r,K,T,S0);
err = abs(V0true - v);
end
