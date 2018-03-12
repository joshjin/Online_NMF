% TODO:  check   1) implement onmf with general diverence as introduced in the paper
%                            2) vectorize equation

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

% Here we set V to be known instead of a stream of input
function [W,H] = addi_update_nmf(V, k)

[m, n] = size(V);
batch_size = 20;

% intialize W and H
W = rand(m,k);
H = rand(k,n);

for t = 1: n/batch_size
    step_size = 10000 / (10000 + t * batch_size);
    vt = V(:, (t-1)*batch_size+1:t*batch_size);
    
    Z2 = ones(m,batch_size);
    
    % Learn the coefficient vector h_t per algorithm 2
    ht0 = rand(k,batch_size) ;
    ht = learning_h_t(ht0, W, vt, step_size, 100, Z2);

    Z1 = vt ./ (W*ht);
    W = W + step_size * (Z1 * ht' - Z2 * ht');
end

function ht = learning_h_t(ht, W, vt, btk, g, Z2)
% this is corresponding to algorithm 2 of the paper
% <Inputs>
%       ht: initial coefficient vector h_t_0
%       Wt1: basis matrix W_(t-1)
%       vt: data sample
%       btk: step size beta_t_k
%       g: maximum number of iterations gama
%<Outputs>
%       ht: final coefficient vector h_t := h_t_gama

% Try multiplicative update rule proposed in Lee & Seung's "Algorithms for Non-negative Matrix Factorization"
% as a good compromise of between speed and ease of implementation.
% Here we can ignore btk for now.

for r = 1:g
    
    Z1 = vt ./ (W*ht);
    ht = ht + btk * (W' * Z1 - W' * Z2);
    
end

end

H = pinv(W) * V;

end











