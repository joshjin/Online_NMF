addpath pcode;
addpath layers;

my_model = load('model_2nd_net.mat');
model = my_model.model;
file = load('trainedH10.mat');
nmf_param = file.nmf_param;

batch_size = 8;
iters = 32;

i = 2;   % set of testing

% read data into var: data
fid = fopen( 't10k-images.idx3-ubyte', 'r' );
data_temp = fread( fid, 'uint8' );
fclose(fid);
data_temp = data_temp(17:end);
data_temp = reshape( data_temp, 28*28, 10000 )';
data_test = data_temp(256*(i-1)+1:256*i,:);

% read label into var: label
fid = fopen( 't10k-labels.idx1-ubyte', 'r' );
label_temp = fread( fid, 'uint8' );
fclose(fid);
label_temp = label_temp(9:end);
label_test = label_temp(256*(i-1)+1:256*i);

H1= nmf_step_in_test(V, nmf_param.W1,26);
H2= nmf_step_in_test(H1', nmf_param.W2,24);
H3= nmf_step_in_test(H2', nmf_param.W3,22);
H4= nmf_step_in_test(H3', nmf_param.W4,20);
H5= nmf_step_in_test(H4', nmf_param.W5,18);
H6= nmf_step_in_test(H5', nmf_param.W6,16);
H7= nmf_step_in_test(H6', nmf_param.W7,14);
H8= nmf_step_in_test(H7', nmf_param.W8,12);
H9= nmf_step_in_test(H8', nmf_param.W9,10);
H10= nmf_step_in_test(H9', nmf_param.W10,8);

[s1, s2] = size(H10);
s0 = 6;
feat_num = 80;
% flaten
% fully connected
% loss function
input = reshape(H10, feat_num, s0, s0, s2/s0/s0); 
input = reshape(input, s0, s0, feat_num, s2/s0/s0);    % new size = 6x6x80x256

overall_right = 0;
for i=1:iters
    batch = input(:,:,:,(i-1)*batch_size+1:i*batch_size);
    batch_label = label_test((i-1)*batch_size+1:i*batch_size,:);
    [output,~] = inference_(model,batch);
    [~,I] = max(output);
    comp = I' == batch_label;
    overall_right = overall_right + sum(comp);
    accuracy = 100 * overall_right / (i*batch_size);
    disp(accuracy);
end