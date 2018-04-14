function [label_mat] = turn_label_to_mat(label)
[len,~] = size(label);
label_mat = zeros(10, len);
for i = 1:len
    label_mat(label(i)+1,i) = 1;
end
disp('label matrix dimension:');
disp(size(label_mat));
end

