%{
  Convergence rate for Finite difference methods vs for Euler antithetic
  to price a European Call Options

  By Alessandro Piccolo, Jim Lindberg and Rui Hao
%}
clear all; close all;

%% General input parameters
K        = 15;             % Strike price
r        = 0.1;            % Interest rate
sigma    = 0.25;           % Diffusion paramter
T        = 0.5;            % Final time
gamma    = 1;              % Elasticity variable

s_min    = 0;              % Min stock price
s_max    = 4*K;            % Max stock price

M        = 1000;           % Number of stock prices (steps)
N        = 10000;          % Number of time steps

Nvec     = 30:5:200;       % Time data points for convergence rate
Mvec     = 10:5:80;        % Stock price data points for convergence rate
lengthNvec = length(Nvec);
lengthMvec = length(Mvec);

%% Extra input parameters for Monte-Carlo method (Euler antithetic)
S0            = 14;        % Initial stock price
nrSimulations = 5;         % Number of Monte-Carlo simualtions
sample_path   = 1e5;       % Number of sample paths <-- changed from 5 000
numN          = [1:500];   % Number of time steps <--- change to 500
numN_length   = length(numN);
errN(numN_length, nrSimulations) = 0;

% No. of sample paths
time_steps = 500; % <-- changed from 200
numM  = [1e1, 50, 1e2, 150, 300, 600, 700, 800, 1e3, 1e4, 1e5,4e5,8e5,1e6];
numM_length = length(numM);
errM(numM_length, nrSimulations) = 0;    % MC antithetic
errM_mc(numM_length, nrSimulations) = 0; % MC

%% FDM: Accuracy, error - Varying time steps values, N
tmp_error_N = zeros(1, M+1);
h = waitbar(0, 'FDM implicit: Loading please wait...');
for i = 1:lengthNvec
    [v,~,~,~,svec] = implicit(K,r,sigma,T,gamma,s_min,s_max,M,Nvec(i));
    for j = 1:M+1
        v_exact        = bsexact(sigma, r, K, T, svec(j));
        tmp_error_N(j) = abs(v(j,1)-v_exact); % Calculate error
    end
    error_mean_N(i) = mean(tmp_error_N);
    waitbar(i/lengthNvec, h);
end
close(h);

p_error_Nvec = polyfit(log10(Nvec),log10(error_mean_N),1); % linear approx

figure(1)
subplot(1,2,1)
plot(log10(Nvec), log10(error_mean_N), 'r*');
l_line = lsline; set(l_line(1),'color','b')
str = 'Implicit: Polyfitted sample error with varying no. of time steps';
hlt = title(str);
hlx = xlabel('No. of time steps');
hly = ylabel('Sample error');
set(hly,'FontSize',13,'FontWeight', 'bold');
set(hlx,'FontSize',13,'FontWeight', 'bold');
set(hlt,'FontSize',13,'FontWeight', 'bold');
legend('FDM implicit','FDM implicit: regression line')
grid on

display(['FDM implicit: Varying time step - Slope = ' ...
    num2str(p_error_Nvec(1))]);

%% Euler Antithetic: Discretization error for varying number of time steps
k = 1;
for time_step = numN
    for i = 1:nrSimulations
        Z = randn(sample_path,time_step);
        [~,errN(k,i)] = mc_antithetic(T,S0,K,r,sigma,gamma,time_step,...
            sample_path,Z);
    end
    k = k+1; 
end

% Polyfit to make linear approximation of data
p_mc_a_time_step = polyfit(log10(numN),log10(mean(errN')),1);

figure(1)
subplot(1,2,2)
plot(log10(numN), log10(mean(errN')), 'r*');
l_line = lsline; set(l_line(1),'color','b')
str = ['Euler Antithetic: Polyfitted sample error ' ...
    'with varying no. of time steps'];
hlt = title(str);
hlx = xlabel('No. of time steps');
hly = ylabel('Sample error');
set(hly,'FontSize',13,'FontWeight', 'bold');
set(hlx,'FontSize',13,'FontWeight', 'bold');
set(hlt,'FontSize',13,'FontWeight', 'bold');
legend('Euler antithetic','Euler   regression line')
grid on

display(['Euler antithetic: Varying time step - Slope = ' ...
    num2str(p_mc_a_time_step(1))]);

%% Accuracy, error - Varying price option steps values, M
for i = 1:lengthMvec
    [v,~,~,~,svec] = implicit(K,r,sigma,T,gamma,s_min,s_max,Mvec(i),N);
    tmp_error_M = zeros(1, Mvec(i)+1);
    for j = 1:Mvec(i)+1
        v_exact        = bsexact(sigma, r, K, T, svec(j));
        tmp_error_M(j) = abs(v(j,1)-v_exact); % Calculate error
    end
    error_mean_M(i) = mean(tmp_error_M);
end

p_error_Mvec = polyfit(log10(Mvec),log10(error_mean_M),1); % linear approx

figure(3)
plot(log10(Mvec), log10(error_mean_M), 'r*');
l_line = lsline; set(l_line(1),'color','b')
str = ['Implicit: Polyfitted sample error with varying' ...
    ' no. of price stock steps'];
hlt = title(str);
hlx = xlabel('Number of stock price steps');
hly = ylabel('Sample error');
set(hly,'FontSize',13,'FontWeight', 'bold');
set(hlx,'FontSize',13,'FontWeight', 'bold');
set(hlt,'FontSize',13,'FontWeight', 'bold');
legend('FDM implicit','FDM:   regression line')
grid on

display(['Varying stock price step - Slope = ' num2str(p_error_Mvec(1))]);

%% Euler antithetic: Sample error for varying number of sample paths
%  And Normal Monte-Carlo (MC)
k = 1;
for sample_paths = numM
    for j = 1:nrSimulations
        Z = randn(sample_paths, time_steps);
        [~,errM(k,j)] = mc_antithetic(T,S0,K,r,sigma,gamma,time_steps,...
            sample_paths,Z);
        [~,errM_mc(k,j)] = mc(T,S0,K,r,sigma,gamma,time_steps, ...
            sample_paths,Z);
    end
    k = k+1;
end

% Polyfit to make linear approximation of data
p_mc_a_sample_path = polyfit(log10(numM),log10(mean(errM')),1);
p_mc_sample_path = polyfit(log10(numM),log10(mean(errM_mc')),1);

figure(4)
subplot(1,2,1)
plot(log10(numM), log10(mean(errM')), 'r*');
l_line = lsline; set(l_line(1),'color','r')
grid on
hold on

display(['Euler antithetic: Varying sample path - Slope = ' ...
    num2str(p_mc_a_sample_path(1))]);

figure(4)
subplot(1,2,2)
plot(log10(numM), log10(mean(errM_mc')), 'b*');
l_line = lsline; set(l_line(1),'color','b')
str = ['Euler MC and MC antithetic: with varying no. of sample paths'];
hlt = title(str);
hlx = xlabel('No. of sample paths');
hly = ylabel('Sample error');
set(hly,'FontSize',13,'FontWeight', 'bold');
set(hlx,'FontSize',13,'FontWeight', 'bold');
set(hlt,'FontSize',13,'FontWeight', 'bold');
grid on

display(['Euler MC (normal): Varying sample path - Slope = ' ...
    num2str(p_mc_sample_path(1))]);
