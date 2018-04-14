load trainedH10.mat;
disp(size(H10));
disp(size(H10',1)/36);
[flatten] = flatten_layer(H10',size_before_flatten,80);
[data_length, im_size] = size(flatten);
f = flatten';
A1 = zeros(10, data_length);
[Z1, dv_input, grad] = linear_layer(flatten, params_linear, 10, 0);
disp('size of Z1');
disp(size(Z1));

layers = [imageInputLayer([2880 1])
          fullyConnectedLayer(144)
          reluLayer
          fullyConnectedLayer(10)
          softmaxLayer
          classificationLayer];

options = trainingOptions('sgdm');

net = trainNetwork(f, layers, options);
