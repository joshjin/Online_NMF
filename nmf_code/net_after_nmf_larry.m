% neural net script after training through 10-layer nmf
addpath pcode;
addpath layers;
addpath tools;

input = load('trainedH10.mat', 'H10');
H10 = input.H10;
disp("H10 size: ");
disp(size(H10));

% flaten
% fully connected
% loss function

l = [
    init_layer('flatten',struct('num_dims',4))
	init_layer('linear',struct('num_in',720,'num_out',10))
	init_layer('softmax',[])
    ];

model = init_model(l,[28 28 1],10,true);

% Learning rate
lr = 0.1;
% Weight decay
wd = .0005;
% Batch size
batch_size = 64;

% Saved model name
save_file = 'model.mat';

params = struct('learning_rate',lr,'weight_decay',wd,'batch_size',batch_size,'save_file',save_file,'epoch',0);

numIters = size(train_data,4) / batch_size;

max_epochs = 5;

for epoch=1:max_epochs
    params.epoch = epoch;
    [model, loss] = train(model,train_data,train_label,params,numIters);
end
    
