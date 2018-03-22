function [W,H] = nmf_step_new(data,patch_size,feat_out)
[data_len,feat_len] = size(data);
data = reshape(data,patch_size,patch_size,data_len/(patch_size*patch_size),feat_len);
V = merge_local_feat(data);
V = V';
batch_size = 64;
[m, n] = size(V);
W = rand(m,feat_out);
H = rand(feat_out,n);
for t = 1: n/batch_size
    vt = V(:, (t-1)*batch_size+1:t*batch_size);
    [W] = onmf(vt, feat_out, W);
    disp(strcat('finished batch #', num2str(t)));
end
% vt = V(:, n/batch_size:n);
% [W] = onmf(vt, feat_out, W);
end

