function [model, loss] = train(model,input,label,params,numIters)

% addpath ../pcode;

% Initialize training parameters
% This code sets default values in case the parameters are not passed in.

% Learning rate
if isfield(params,'learning_rate') lr = params.learning_rate;
else lr = .01; end
% Weight decay
if isfield(params,'weight_decay') wd = params.weight_decay;
else wd = .0005; end
% Batch size
if isfield(params,'batch_size') batch_size = params.batch_size;
else batch_size = 128; end

%By default the code saves the model in 'model.mat'
if isfield(params,'save_file') save_file = params.save_file;
else save_file = 'model_2nd_net.mat'; end

% update_params will be passed to your update_weights function.
% This allows flexibility in case you want to implement extra features like momentum.
update_params = struct('learning_rate',lr,'weight_decay',wd);

% file_name = ['result1.txt'
%     'result2.txt'
%     'result3.txt'
%     'result4.txt'
%     'result5.txt'];
% 
% file = file_name(params.epoch,:);
% fid = fopen(file,'w');

overall_right = 0;


for i = 1:numIters
	% TODO: Training code
    % The basic loop goes like this:
    %   Select a subset of your dataset to be a batch
    %   Run inference
    %   Calculate your loss
    %   Calculate your gradients
    %   Update the weights of your model
    %   Repeat
    batch = input(:,:,:,(i-1)*batch_size+1:i*batch_size);
    batch_label = label((i-1)*batch_size+1:i*batch_size,:);
    [output,activations] = inference_(model,batch);
    
    [~,I] = max(output);
    comp = I' == batch_label;
    overall_right = overall_right + sum(comp);
    accuracy = 100 * overall_right / (i*batch_size);
    
    [loss, dv_input] = loss_crossentropy(output, batch_label, update_params, true);
    [grad] = calc_gradient_(model, batch, activations, dv_input);
    model = update_weights_(model,grad,update_params);
    if mod(i, 10)==0
        X=sprintf('EPOCH:   %d   ITER:   %d   LOSS:   %1.2d\n',params.epoch,i,loss);
        disp(X);
        disp(accuracy);
%         fprintf(fid, X);
    end
end
save(save_file,'model');
% fid = fclose(fid);