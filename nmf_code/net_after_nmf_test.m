addpath pcode;
addpath layers;

my_model = load('model_2nd_net.mat');
model = my_model.model;
file = load('trainedH10.mat');
nmf_param = file.nmf_param;

batch_size = 8;
iters = 32;

i = 1;   % set of testing

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
label(label==0)=10;

nV = expand3x3(data_test);
[len,~,~] = size(nV);
nV = reshape(nV,len,9);
% nV = nV + (V < 0.95) * 1e-3;

nH1= nmf_step_in_test(nV, nmf_param.W1,26);
% disp('diff1: ');
% disp(abs(nH1-H1));
% disp(max(max(abs(nH1-H1))));
nH2= nmf_step_in_test(nH1', nmf_param.W2,24);
nH3= nmf_step_in_test(nH2', nmf_param.W3,22);
nH4= nmf_step_in_test(nH3', nmf_param.W4,20);
nH5= nmf_step_in_test(nH4', nmf_param.W5,18);
nH6= nmf_step_in_test(nH5', nmf_param.W6,16);
nH7= nmf_step_in_test(nH6', nmf_param.W7,14);
nH8= nmf_step_in_test(nH7', nmf_param.W8,12);
nH9= nmf_step_in_test(nH8', nmf_param.W9,10);
nH10= nmf_step_in_test(nH9', nmf_param.W10,8);

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