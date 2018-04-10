function [W,H]=nmf_step(data,patch_size,feat_out)
disp(size(data));
[data_len,feat_len] = size(data);
disp(size(data));
data = reshape(data,patch_size,patch_size,data_len/(patch_size*patch_size),feat_len);
disp(size(data));
V = merge_local_feat(data);
disp(size(V));
[W,H] = nnmf(V',feat_out);

end