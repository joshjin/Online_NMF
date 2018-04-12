addpath pcode;
addpath layers;
%  not yet done
l = [init_layer('nmf',struct('feat_no',20))
%     init_layer('relu',[])
%     init_layer('pool',struct('filter_size',5,'stride',2))
    init_layer('nmf',struct('feat_no',20))
%     init_layer('relu',[])
%     init_layer('pool',struct('filter_size',3,'stride',1))
    init_layer('flatten',struct('num_dims',4))
	init_layer('linear',struct('num_in',720,'num_out',10))
	init_layer('softmax',[])];

model = init_model(l,[28 28 1],10,true);

% Example calls you might make:
% [output,~] = inference(model,input);
% [loss,~] = loss_euclidean(output,ground_truth,[],false);

% Learning rate
lr = 0.1;
% Weight decay
wd = .0005;
% Batch size
batch_size = 120;

% Saved model name
save_file = 'model.mat';

params = struct('learning_rate',lr,'weight_decay',wd,'batch_size',batch_size,'save_file',save_file,'epoch',0);

numIters = size(train_data,4) / batch_size;

max_epochs = 5;

for epoch=1:max_epochs
    params.epoch = epoch;
    [model, loss] = train(model,train_data,train_label,params,numIters);
end