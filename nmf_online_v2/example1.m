% ------------------------------ Usage examples ------------------------------
m = 784;
n = 100;
k = 20;

A = rand(m,n);

[W,H] = onmf_batch(A,k);

[W2, H2] = nnmf(A, k);

% disp(A - W * H);

% ----------------------------- DIfference Check --------------------------
d = A - W*H;
% max(d)
% max(d(:))
% size(d)
% min(d(:))
% d = A - W*H;
% max(d(:))
% min(d(:))
d = abs(d);
% max(d(:))
% hist(d(:), 100);
% sum(d > 1)
disp("percentage of entries diff > .5: ");
disp(sum(d(:) > .5 ) / (m*n) * 100);
% 8320 / 200
% 8320 / 200/300

d2 = A - W2*H2;
d2 = abs(d2);
disp("percentage of nnmf entries diff > .5: ");
disp(sum(d2(:) > .5) / (m*n) * 100);