function [W,H]=nmf_step(data,patch_size,feat_out)
[data_len,feat_len] = size(data);
data = reshape(data,patch_size,patch_size,data_len/(patch_size*patch_size),feat_len);
V = merge_local_feat(data);
[W,H] = nnmf(V',feat_out);

end