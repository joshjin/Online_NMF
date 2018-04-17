function H=nmf_step_in_test(data, W,patch_size)
disp('size of the input data for nmf_step');
disp(size(data));
[data_len,feat_len] = size(data);
data = reshape(data,patch_size,patch_size,data_len/(patch_size*patch_size),feat_len);
disp('reshaped data');
disp(size(data));
V = merge_local_feat(data);
disp('size of V after merge feature');
disp(size(V));
H = W' * V';
end