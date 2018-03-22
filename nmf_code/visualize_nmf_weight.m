function visualize_nmf_weight(weight_list,total)
wlen = length(weight_list);
PATCH_SIZE = 3 + wlen*2;
inv_weights = cell(1,wlen);
for i = 1:wlen
    [~,wy] = size(weight_list{i});
    inv_weights{i} = eye(wy,wy)/weight_list{i};
end
%[wx,wy] = size(weight);
%weight = eye(wy,wy)/weight;
img = zeros(PATCH_SIZE*PATCH_SIZE+(PATCH_SIZE-1)*1,PATCH_SIZE*PATCH_SIZE+(PATCH_SIZE-1)*1);
for i = 1:5
    for j = 1:5
        if (i-1)*5+j >  total
            break
        end
        %f = reshape(weight(:,(i-1)*5+j),3,3,3,3,1);
        %f = shrink_neighbor(f);
        buffer = zeros(PATCH_SIZE,PATCH_SIZE);
        f = recurr_collapse_nmf(buffer,fix(PATCH_SIZE/2)+1,fix(PATCH_SIZE/2)+1,(i-1)*5+j,inv_weights);
        %f = (f - min(min(f)))/(max(max(f))-min(min(f)));
        img(1+(i-1)*(PATCH_SIZE+1):PATCH_SIZE + (i-1)*(PATCH_SIZE+1),1+(j-1)*(PATCH_SIZE+1):PATCH_SIZE + (j-1)*(PATCH_SIZE+1)) = f;
    end
end
imshow(img,[]);
end