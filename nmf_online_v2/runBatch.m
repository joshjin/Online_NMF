addpath('/..');

mnist_data = load_MNIST_images('t10k-images.idx3-ubyte');

k = 5;
sample_size = 100;

V = mnist_data(:, 1:sample_size)+1e-3;

%[W, H] = RPMF(V, k, 1, 1, 1e-6);
[W2, H2] = onlineRPMF(V, k, 1, 1, 1e-6, ones(size(V)));
disp('nmf is done');

mkdir('mnist_features') ;

V_new = W * H;
V_new2 = W2 * H2;

d = V - V_new;
d2 = V - V_new2;

disp("RPMF");
disp(sum(d(:) > .01) / (784 * sample_size) * 100);
disp("onlineRPMF");
disp(sum(d2(:) > .01) / (784 * sample_size) * 100);

for i = 1:20
    ori = reshape(V(:,i), [28, 28]);
    pic = reshape(V_new(:,i), [28, 28]);
    pic2 = reshape(V_new2(:,i), [28, 28]);
    
    imwrite(ori, sprintf('mnist_features/f_%d_original.jpg', i'));
    imwrite(pic, sprintf('mnist_features/f_%d_RPMF.jpg', i'));
    imwrite(pic2, sprintf('mnist_features/f_%d_oRPMF.jpg', i'));
end

for i = 1:k
    imwrite(reshape(W(:, i), [28, 28]), sprintf('mnist_features/Z_basis_batch_%d.jpg', i'));
    imwrite(reshape(W2(:, i), [28, 28]), sprintf('mnist_features/Z_o_basis_batch_%d.jpg', i'));
end
