m = 300;
n = 200;
k = 15;
diff = 0.3;

V = rand(m, n);

[W, H] = addi_update_nmf(V, k);
[W2, H2] = nnmf(V, k);

% disp(V - W * H);

disp(sum (sum (abs(V - W*H) > diff)) / m / n * 100);
disp(sum (sum (abs(V - W2*H2) > diff)) / m / n * 100);