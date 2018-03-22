function dump_weights(weights,version)
n = length(weights);
prev_flen = 9;
for i = 1:n
    [~,ylen] = size(weights{i});
    inv_w = eye(ylen) / weights{i};
    weights{i} = reshape(inv_w,[],3,3,prev_flen);
    [prev_flen,~,~,~] = size(weights{i});
end
save(['weights_v',num2str(version),'.mat'],'weights');
end