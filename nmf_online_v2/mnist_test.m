addpath('/..');

mnist_data = load_MNIST_images('t10k-images.idx3-ubyte');

k = 15;
sample_size = 100;

V = mnist_data(:, 1:sample_size)+1e-3;

% V = rand(784, 100) * 255;
[W, H] = onmf_batch(V,k);
% [W, H] = onmf(V,k);
[W2, H2] = nnmf(V, k);
% [W3, H3] = multi_update_nmf(V, k);
[W3, H3] = multi_update_nmf(V, k);
disp('nmf is done');

mkdir('mnist_features') ;

V_new = W * H;
V_new2 = W2 * H2;
V_new3 = W3 * H3;

d = V - V_new;
d2 = V - V_new2;
d3 = V - V_new3;

disp("batch");
disp(sum(d(:) > .01) / (784 * sample_size) * 100);
disp("nnmf");
disp(sum(d2(:) > .01 ) / (784 * sample_size) * 100);
disp("multi");
disp(sum(d3(:) > .01) / (784 * sample_size) * 100);

for i = 1:20
    ori = reshape(V(:,i), [28, 28]);
    pic = reshape(V_new(:,i), [28, 28]);
    pic2 = reshape(V_new2(:,i), [28, 28]);
    pic3 = reshape(V_new3(:,i), [28, 28]);
    
    imwrite(ori, sprintf('mnist_features/f_%d_original.jpg', i'));
    imwrite(pic, sprintf('mnist_features/f_%d_onmf.jpg', i'));
    imwrite(pic2, sprintf('mnist_features/f_%d_nnmf.jpg', i'));
    imwrite(pic3, sprintf('mnist_features/f_%d_m_u_nmf.jpg', i'));
end

for i = 1:k
    imwrite(reshape(W(:, i), [28, 28]), sprintf('mnist_features/Z_basis_batch_%d.jpg', i'));
    imwrite(reshape(W2(:, i), [28, 28]), sprintf('mnist_features/Z_basis_nnmf_%d.jpg', i'));
    imwrite(reshape(W3(:, i), [28, 28]), sprintf('mnist_features/Z_basis_multi_%d.jpg', i'));
end
