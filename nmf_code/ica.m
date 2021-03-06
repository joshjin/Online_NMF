%% PCA demo -- digit
clear

addpath('package/FastICA_25')

size_before_flatten = 6;
params_linear_1 = struct('W', rand(144,2880), 'b', rand(144,1));
params_linear_2 = struct('W', rand(10,144), 'b', rand(10,1));

% read data into var: data
fid = fopen( 't10k-images.idx3-ubyte', 'r' );
data = fread( fid, 'uint8' );
fclose(fid);
data = data(17:end);
data = reshape( data, 28*28, 10000 )';
short_data = data(1:256,:);
disp('data size');
disp(size(data));
disp('short data size');
disp(size(short_data));

% read label into var: label
fid = fopen( 't10k-labels.idx1-ubyte', 'r' );
label = fread( fid, 'uint8' );
fclose(fid);
label = label(9:end);
disp('label size');
disp(size(label));
label_mat = turn_label_to_mat(label);

show_data = reshape(data,10000,28,28);
disp('show data size');
disp(size(show_data));

disp('Expanding Images to 3x3 Patch');
% V = expand3x3(data);
V = expand3x3(short_data);
[len,~,~] = size(V);
V = reshape(V,len,9);
V = V + (V < 0.95) * 1e-3;

[icasig, A, W] = ica_step(V,26,20);
    
disp(size(A));
disp(size(W));
disp(size(icasig));

% disp('NNMF Optimizing Step 1')
% [W1,H1] = nmf_step(V,26,20);
% disp('NNMF Optimizing Step 2')
% [W2,H2] = nmf_step(H1',24,20);
% disp('NNMF Optimizing Step 3')
% [W3,H3] = nmf_step(H2',22,20);
% disp('NNMF Optimizing Step 4')
% [W4,H4] = nmf_step(H3',20,40);
% disp('NNMF Optimizing Step 5')
% [W5,H5] = nmf_step(H4',18,40);
% disp('NNMF Optimizing Step 6')
% [W6,H6] = nmf_step(H5',16,40);
% disp('NNMF Optimizing Step 7')
% [W7,H7] = nmf_step(H6',14,80);
% disp('NNMF Optimizing Step 8')
% [W8,H8] = nmf_step(H7',12,80);
% disp('NNMF Optimizing Step 9')
% [W9,H9] = nmf_step(H8',10,80);
% disp('NNMF Optimizing Step 10')
% [W10,H10] = nmf_step(H9',8,80);
% 
