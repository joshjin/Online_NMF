function [output, dv_input, grad] = linear_layer(input, params, num_out, backprop)

batch_size = size(input,1);
output = zeros(num_out, batch_size);
output = params.W * input + params.b;

if backprop
	dv_input = zeros(size(input));
	grad.W = zeros(size(params.W));
	grad.b = zeros(size(params.b));
	% TODO: BACKPROP CODE
    dv_input = params.W' * dv_output;
    grad.W = dv_output * input';
    grad.b = dv_output * ones([20, 1]);

end
