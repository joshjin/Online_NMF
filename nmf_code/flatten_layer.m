function [result] = flatten_layer(input, patch_size, feat)
[dim, ~] = size(input);
data_size = dim/(patch_size*patch_size);
result = reshape(input,data_size,patch_size*patch_size*feat);
disp(size(result));
end
