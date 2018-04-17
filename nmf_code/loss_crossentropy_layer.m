function [loss, dv_input] = loss_crossentropy_layer(input, labels, backprop)

disp('size of input');
disp(size(input));
disp('size of labels');
disp(size(labels));
% assert(max(labels) <= size(input,1));

% TODO: CALCULATE LOSS
% if isfield(hyper_params, 'num_dims') num_dims = hyper_params.num_dims;
% else num_dims = 2; end
batch_size = size(input, 2);
% new_labels = zeros([batch_size, size(input,1)]);
% for i = 1:batch_size
%     new_labels(i,labels(i)) = 1;
% end
labels = labels';

loss = 0;
for i = 1:batch_size
    disp('labels(1,:)');
    disp(size(labels(1,:)));
    disp('log(input(:,i)');
    disp(size(log(input(:,i))));
    loss = loss + labels(i,:) * log(input(:,i));
end
loss = - loss / batch_size;

disp('loss');
disp(loss);

dv_input = zeros(size(input));
if backprop
	% TODO: BACKPROP CODE
    divisor = - batch_size;
    log_part = ones(size(input)) ./ input;
    for i = 1:batch_size
        dv_input(:,i) = (labels(i,:) .* log_part(:,i)')';
    end
    dv_input = dv_input / divisor;
    disp('dv_input');
    disp(size(dv_input));
end

