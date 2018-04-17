% neural net script after training through 10-layer nmf
addpath pcode;
addpath layers;
addpath tools;

file = load('trainedH10.mat', 'H10');
H10 = file.H10;
disp("H10 size: ");
disp(size(H10));
[s1, s2] = size(H10);
s0 = 6;
feat_num = 80;
% flaten
% fully connected
% loss function
input = reshape(H10, feat_num, s0, s0, s2/s0/s0); 
input = reshape(input, s0, s0, feat_num, s2/s0/s0);    % new size = 6x6x80x256
l = [
    init_layer('flatten',struct('num_dims',4))
	init_layer('linear',struct('num_in',s0*s0*feat_num,'num_out',10))
	init_layer('softmax',[])
    ];

% Learning rate
lr = 0.5;
% Weight decay
wd = .0003;
% Batch size
batch_size = 8;

model = init_model(l,[s0 s0 feat_num],10,true);



% Saved model name
save_file = 'model_2nd_net.mat';

params = struct('learning_rate',lr,'weight_decay',wd,'batch_size',batch_size,'save_file',save_file,'epoch',0);

numIters = size(input,4) / batch_size;

max_epochs = 50;

% read label into var: label
fid = fopen( 't10k-labels.idx1-ubyte', 'r' );
label = fread( fid, 'uint8' );
fclose(fid);
label = label(9:end);
label(label==0)=10;

for epoch=1:max_epochs
    params.epoch = epoch;
    [model, loss] = train(model,input,label,params,numIters);
end
    
