function result = merge_local_feat(data)
[patch_size,~,data_len,feat_len] = size(data);
result = zeros(data_len * (patch_size - 2) * (patch_size - 2), feat_len * 9);
for idx = 0:data_len-1
    for i = 0:patch_size-3
        for j = 0:patch_size-3
            result(1 + idx * (patch_size - 2) * (patch_size -2) + i * (patch_size - 2) + j,:) = ...
                reshape(data(i+1:i+3,j+1:j+3,idx+1,:),1,feat_len*9);
        end
    end
end
end