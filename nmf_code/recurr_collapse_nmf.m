function c_inv_w = recurr_collapse_nmf(c_inv_w,cx,cy,multiplier,inv_w_list)
PATCH_SIZE = 3;
if length(inv_w_list) == 0
    c_inv_w(cx-1:cx+1,cy-1:cy+1) = c_inv_w(cx-1:cx+1,cy-1:cy+1) + reshape(multiplier,PATCH_SIZE,PATCH_SIZE);
    return;
end
w = inv_w_list{1};
if isscalar(multiplier)
    w = w(multiplier,:);
else
    w = multiplier * w;
end
w = reshape(w,PATCH_SIZE,PATCH_SIZE,[]);
for i = -1:1
    for j = -1:1
        c_inv_w = recurr_collapse_nmf(c_inv_w,cx+i,cy+j,reshape(w(2+i,2+j,:),1,[]),inv_w_list(2:end));
    end
end
end