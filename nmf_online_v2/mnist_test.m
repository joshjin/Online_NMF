addpath('/..');

mnist_data = load_MNIST_images('t10k-images.idx3-ubyte');

V = mnist_data(:, 1:20)+1e-10;

% V = rand(784, 100) * 255;
[W, H] = onmf(V,5);
[W2, H2] = nnmf(V, 5);
disp('onmf is done');

mkdir('mnist_features') ;
% for i = 1:25
%     feature = reshape(W(:,i), [28, 28]);
%     feature2 = reshape(W2(:,i), [28, 28]);
%     imwrite(feature, sprintf('mnist_features/f_%d.jpg', i'));
%     imwrite(feature2, sprintf('mnist_features/g_%d.jpg', i'));
% end

V_new = W * H;
V_new2 = W2 * H2;
for i = 1:20
    ori = reshape(V(:,i), [28, 28]);
    pic = reshape(V_new(:,i), [28, 28]);
    pic2 = reshape(V_new2(:,i), [28, 28]);
    imwrite(ori, sprintf('mnist_features/f_%d_original.jpg', i'));
    imwrite(pic, sprintf('mnist_features/f_%d_onmf.jpg', i'));
    imwrite(pic2, sprintf('mnist_features/f_%d_nnmf.jpg', i'));
end