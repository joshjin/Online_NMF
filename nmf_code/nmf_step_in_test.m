function H=nmf_step_in_test(data, W ,patch_size)

[data_len,feat_len] = size(data);
data = reshape(data,patch_size,patch_size,data_len/(patch_size*patch_size),feat_len);

V = merge_local_feat(data);

H = pinv(W) * V';
end