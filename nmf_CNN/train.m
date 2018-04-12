%--------------------------------------------------------------------------
% main method: 
%--------------------------------------------------------------------------

clear

% read data into var: data
fid = fopen( 't10k-images.idx3-ubyte', 'r' );
data = fread( fid, 'uint8' );
fclose(fid);
data = data(17:end);
data = reshape( data, 28*28, 10000 )';
short_data = data(1:512,:);

% display data size
disp('data size');
disp(size(data));
disp('short data size');
disp(size(short_data));

% run image with 5X5 filter
V = expand5X5(short_data);
disp('filtered size');
disp(size(V));
[len,~,~] = size(V);
V = reshape(V,len,25);




