function [loss, dv_input] = loss_crossentropy_layer(input, labels, backprop)

assert(max(labels) <= size(input,1));

% TODO: CALCULATE LOSS
% if isfield(hyper_params, 'num_dims') num_dims = hyper_params.num_dims;
% else num_dims = 2; end
% batch_size = size(input, num_dims);
% new_labels = zeros([batch_size, size(input,1)]);
% for i = 1:batch_size
%     new_labels(i,labels(i)) = 1;
% end

loss = 0;
for i = 1:batch_size
    loss = loss + new_labels(i,:) * log(input(:,i));
end
loss = - loss / batch_size;

dv_input = zeros(size(input));
if backprop
	% TODO: BACKPROP CODE
    divisor = - batch_size;
    log_part = ones(size(input)) ./ input;
    for i = 1:batch_size
        dv_input(:,i) = (new_labels(i,:) .* log_part(:,i)')';
    end
    dv_input = dv_input / divisor;

end

