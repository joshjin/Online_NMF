function result=reconstruct_step(W,H,image_size)
[~,dlen] = size(H);
dlen = dlen/image_size/image_size;
[flen,~] = size(W);
Hpre = W * H;
Hpre = reshape(Hpre,flen,image_size,image_size,dlen);
Hpre = reshape(Hpre,3,3,flen/9,image_size,image_size,dlen);
result = zeros(flen/9,image_size+2,image_size+2,dlen);
for i = 1:3
    for j = 1:3
        result(:,i:i+image_size-1,j:j+image_size-1,:) = result(:,i:i+image_size-1,j:j+image_size-1,:) + reshape(Hpre(i,j,:,:,:,:)/9,flen/9,image_size,image_size,dlen);
    end
end
result=reshape(result,flen/9,[]);
end