load trainedH10.mat;
% disp(size(H10));
% disp(size(H10',1)/36);
% [flatten] = flatten_layer(H10',size_before_flatten,80);
% [data_length, im_size] = size(flatten);
% f = flatten';
% A1 = zeros(10, data_length);
% [Z1, dv_input, grad] = linear_layer(flatten, params_linear, 10, 0);
% disp('size of Z1');
% disp(size(Z1));
% 
% layers = [imageInputLayer([2880 1])
%           fullyConnectedLayer(144)
%           reluLayer
%           fullyConnectedLayer(10)
%           softmaxLayer
%           classificationLayer];
% 
% options = trainingOptions('sgdm');
% 
% net = trainNetwork(f, layers, options);

disp(size(H10',1)/36);
[flatten] = flatten_layer(H10',size_before_flatten,80);
[data_length, im_size] = size(flatten);
% TODO: forward propagation (not debuged)
[Z1, ~] = linear_layer(flatten, params_linear_1, 144, 0, 0);
disp('size of Z1');
disp(size(Z1));
[A1, ~] = relu_layer(Z1, 0, 0);
disp('size of A1');
disp(size(A1));
[Z2, ~] = linear_layer(A1', params_linear_2, 10, 0, 0);
disp('size of Z2');
disp(size(Z2));
[A2, ~] = relu_layer(Z2, 0, 0);
disp('size of A2');
disp(size(A2));
A3 = softmax(A2);
% TODO: loss_f
[loss, dv_input] = loss_crossentropy_layer(A3, label_mat(:,1:256), 1);


