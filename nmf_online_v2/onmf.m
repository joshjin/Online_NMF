% TODO: implement onmf with general diverence as introduced in the paper:
% Online Nonnegative Matrix Factorization with General Divergences

% <Inputs>
%        V : Input data matrix (m x n)
%        k : Target low-rank
%
%        (Below are parameters pre-defined)
%        
%        MAX_ITER : Maximum number of iterations. Default is 100.
%        MIN_ITER : Minimum number of iterations. Default is 20.
%        MAX_TIME : Maximum amount of time in seconds. Default is 100,000.
%        W_INIT : (m x k) initial value for W.
%        H_INIT : (k x n) initial value for H.
%        TOL : Stopping tolerance. Default is 1e-3. If you want to obtain a more accurate solution, decrease TOL and increase MAX_ITER at the same time.
% <Outputs>
%        W : Obtained basis matrix (m x k)
%        H : Obtained coefficients matrix (k x n)

function [W,H] = onmf(V, k)

[m, n] = size(V);

% Default configuration
par.m = m;
par.n = n;
par.max_iter = 100;
par.min_iter = 20;
par.max_time = 1e6;
par.tol = 1e-3;

% intialize W and H
W = rand(m,k);
H = rand(k,n);

for t = 1: max_iter
    % draw a data sample v_t from P
    v_t = V(:, t);
    
    % Learn the coefficient vector h_t per algorithm 2
    
    
    % Update the basis matrix from W_t-1 to W_t
    
end

function ht = learning_h_t(ht0, Wt1, vt, btk, g)
% this is corresponding to algorithm 2 of the paper
% <Inputs>
%       ht0: initial coefficient vector h_t_0
%       Wt1: basis matrix W_(t-1)
%       v_t: data sample
%       btk: step size beta_t_k
%       g: maximum number of iterations gama
%<Outputs>
%       ht: final coefficient vector h_t := h_t_gama

for k = 1:g
    
end













