%--------------------------------------------------------------------------
% expand original image by 5X5 filter
%--------------------------------------------------------------------------

function [ result ] = expand5X5( data_mat )
[len,~] = size(data_mat);
disp(len);
result = zeros(len * 24 * 24, 5, 5);
data_mat = reshape(data_mat,len,28,28);

for idx = 0:len-1
    for i = 0:28-5
        for j = 0:28-5
            result(1 + idx * 24 * 24 + i * (28 - 4) + j,:,:) = data_mat(1+idx,i+1:i+5,j+1:j+5);
        end
    end
end


end

