% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% output: num_nodes x batch_size
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_softmax(input, params, hyper_params, backprop, dv_output)

[num_classes,batch_size] = size(input);
output = zeros(num_classes, batch_size);
% TODO: FORWARD CODE
divisor = sum(exp(input));
for i=1:num_classes
    output(i,:) = exp(input(i,:)) ./ divisor;
end

dv_input = [];

% This is included to maintain consistency in the return values of layers,
% but there is no gradient to calculate in the softmax layer since there
% are no weights to update.
grad = struct('W',[],'b',[]); 

if backprop
	dv_input = zeros(size(input));
	% TODO: BACKPROP CODE
    %deriv = zeros(num_classes, batch_size);
    dydxi = zeros(num_classes, batch_size);
    for i=1:num_classes
        for j=1:num_classes
            if i==j
                dydxi(j,:) = output(i,:) .* (1 - output(i,:));
            else
                dydxi(j,:) = - output(i,:) .* output(j,:);
            end
        end
        dv_input(i,:) = sum(dydxi .* dv_output);
    end
end
