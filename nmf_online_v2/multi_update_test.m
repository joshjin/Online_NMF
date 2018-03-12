V = rand(300, 200);
k = 15;

[W, H] = multi_update_nmf(V, k);

% disp(V - W * H);

disp(sum (sum (abs(V - W*H) > 0.5)) / 300 / 200);