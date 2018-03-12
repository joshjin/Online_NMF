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
function [W,H] = multi_update_nmf(V, k)

[m, n] = size(V);

% Default configuration
par.m = m;
par.n = n;
par.max_iter = 100;
par.max_time = 1e6;
par.tol = 1e-3;

% intialize W and H
W = rand(m,k);
H = rand(k,n);

for t = 1: 100
    
%     for u = 1:size(H, 2)
%         for a = 1:size(H,1)
%             temp = W * H;
%             top = sum (W(:,a) .* V(:,u) ./ temp(:,u));
%             H(a,u) = H(a,u) * (top / sum(W(:,a)));
%         end
%     end
%    
%     
%     % Update the basis matrix from W_t-1 to W_t
%     for i = 1:size(W, 1)
%         for a = size(W, 2)
%             temp = W * H;
%             top = sum (H(a,:) .* V(i,:) ./ temp(i,:));
%             W(i,a) = W(i,a) * ( top / sum(H(a,:), 2));
%         end
%     end
    
%     temp = W * ht;
%     for i = 1:784
%         for a = 1:20
%             W(i,a) = W(i,a) * (ht(a)*vt(i) / temp(i)) / ht(a);
%         end
%     end

% Eclidean distance multiplicative rule
    temp1 = (W' * V) ./ (W' * W * H);
    H = H .* temp1;
    
    temp2 = (V * H') ./ (W * H * H');
    W = W .* temp2;
    
end

% disp(acc / 784);
% H = W' * V;

end












