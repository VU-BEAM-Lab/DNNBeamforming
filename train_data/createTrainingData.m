% Copyright 2020 Jaime Tierney, Adam Luchies, and Brett Byram

% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the license at

% http://www.apache.org/licenses/LICENSE-2.0

% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and 
% limitations under the License.

function createTrainingData(fullfileIN,fullfileOUT,crange,inputType,...
                k_analyze,N_window)

% INPUTS TO FUNCTION
% fullfileIN: full file path to delayed RF channel data
% fullfileOUT: fullfile path to output h5 file
% crange: axial crop range
% inputType: 'single' or 'multi'
% k_analyze: indicies of depths to analyze within axial kernel
% N_window: axial kernel size in samples
%
% NOTE: the input channel data mat file should have the following
% parameters: 
% 
% t: axial sample indicies
% fs: sampling frequency (MHz)
% c: sound speed (m/s)
% beam_position_x: lateral beam positions (m)
% cyst_center: axial and lateral location of cyst in meters
% cyst_radii: diameter of cyst and background regions for accept/reject

%% LOAD IN AND REFORMAT DELAYED CHANNEL DATA

% load chandat
load(fullfileIN);

% crop channel data to training depths
start = find(t/fs*c/2 >= crange(1), 1, 'first');
stop = find(t/fs*c/2 <= crange(2), 1, 'last');
chandat = chandat(start:stop,:,:);
t = t(start:stop);

% get size of the channel data
[M, N_elements, N_beams] = size(chandat);

% get analytic data
chandat_hilbert = zeros(size(chandat));
for k=1:N_beams
    chandat_hilbert(:,:,k)=hilbert(chandat(:,:,k));
end

% get envelope data for input and target data
env = abs(hilbert(squeeze(sum(chandat,2))));
env = env./max(env(:));
clear chandat 

% save start offsets 
N = N_window;
fracOvrlp = 0.9;
if round(N*fracOvrlp)==N,dz=1;else,dz=N-round(N*fracOvrlp);end
startOffsets = 1:dz:(M-N+1);

% reshape data to add kernel dimension 
znew = zeros(N,length(startOffsets),N_elements,N_beams);
for k=1:length(startOffsets)
    znew(:,k,:,:) = chandat_hilbert(startOffsets(k):startOffsets(k)+N-1,:,:);
end
clear chandat_hilbert

% save chandat as single precision pytorch tensors
old_dat_real = single( real(znew) );
old_dat_imag = single( imag(znew) );
clear znew 

% pull out depths to use within axial kernel
old_dat_real = old_dat_real(k_analyze, :, :, :);
old_dat_real = permute(old_dat_real, [2, 4, 3, 1]);
old_dat_imag = old_dat_imag( k_analyze, :, :, :);
old_dat_imag = permute(old_dat_imag, [2, 4, 3, 1]);

%% SPLIT TRAINING EXAMPLES INTO ACCEPT AND REJECT REGIONS

% get sizing info
N_kernels = size(old_dat_real, 1);
N_beams = size(old_dat_real, 2);
N_elements = size(old_dat_real, 3);
N_depths = size(old_dat_real, 4);

% create depth vector
depth = (t / fs) * c / 2;

% create grid for beam data
X = zeros(length(depth), length(beam_position_x));
Z = zeros(length(depth), length(beam_position_x));
for nbeam = 1:length(beam_position_x)
    X(:, nbeam) = beam_position_x(nbeam) * ones(length(depth), 1);
    Z(:, nbeam) = depth;
end
X = X(startOffsets,:);
Z = Z(startOffsets,:);

% create circular mask for inside lesion
radius_in = cyst_radii(1);
xc = cyst_center(1);
zc = cyst_center(2);
mask_in = sqrt((X-xc).^2 / radius_in^2 + (Z-zc).^2 / radius_in^2) <= 1;

% create circular masks for outside of lesion
radius_out_0 = cyst_radii(2);
A_in = pi * radius_in^2;
radius_out_1 = sqrt( (3*A_in + pi*radius_out_0^2)/pi );
mask_out_0 = sqrt((X-xc).^2 / radius_out_0^2 + (Z-zc).^2 / radius_out_0^2) <= 1;
mask_out_1 = sqrt((X-xc).^2 / radius_out_1^2 + (Z-zc).^2 / radius_out_1^2) <= 1;
mask_out = mask_out_1 - mask_out_0;
    
% define accept and reject regions
select_segments = zeros(N_kernels, N_beams);
select_segments(mask_in(:)==1)=2;
select_segments(mask_out(:)==1)=1;

% reshape things
old_dat_real = reshape(old_dat_real, [], N_elements, N_depths);
old_dat_imag = reshape(old_dat_imag, [], N_elements, N_depths);

%% MAKE QC FIGURES

figure('Visible', 'off');
imagesc(beam_position_x*1000,depth*1000,20*log10(env), [-60, 0]), colormap gray
axis image
xlabel('Lateral Position (mm)');
ylabel('Depth (mm)');
print(strrep(fullfileOUT,'.h5','.png'), '-dpng')
mask_y_all = repmat(depth(startOffsets)*1000, 1, N_beams);
mask_x_all = repmat(beam_position_x*1000, N_kernels, 1);
mask_y_all_out = mask_y_all(select_segments == 1);
mask_x_all_out = mask_x_all(select_segments ==1 );
mask_y_all_in = mask_y_all(select_segments == 2);
mask_x_all_in = mask_x_all(select_segments ==2 );
hold on
scatter(mask_x_all_out(:), mask_y_all_out(:), 5,[102,166,30]./255,'filled')
scatter(mask_x_all_in(:), mask_y_all_in(:), '.r'), hold off
print(strrep(fullfileOUT,'.h5','_selectedSegments.png'), '-dpng')

% save select segments for QC
save(strrep(fullfileOUT,'.h5','_selectedSegments.mat'),'select_segments')
select_segments = reshape(select_segments, N_beams*N_kernels, 1);

%% SET UP INPUT X AND TARGET Y FOR DNNS

% create X and Y data
idx_in_all = find(select_segments==2);
idx_out_all = find(select_segments==1);
idx_all = union(idx_in_all,idx_out_all);
select_segments_nonzero = select_segments(select_segments>0);
idx_in = find(select_segments_nonzero==2);
X = cat(2, old_dat_real(idx_all, :, :), old_dat_imag(idx_all, :, :));
Y = X; 
Y(idx_in,:,:)=0; % NOTE: THIS SCRIPT ASSUMES ANECHOIC CYSTS

% normalize the data
if strcmp(inputType,'multi')
    X = reshape(X,size(X,1),size(X,2)*size(X,3));
    Y = reshape(Y,size(Y,1),size(Y,2)*size(Y,3));
end
X_l1 = max( abs(X), [], 2);
X_l1 = repmat(X_l1, 1, size(X,2), 1);
X = X ./ X_l1;
Y = Y ./ X_l1;
clear X_l1

% shuffle the data
k = randperm(size(X, 1));
X = X(k, :, :);
Y = Y(k, :, :);

% set size of the dataset
disp(size(X));
N_train = size(X, 1);
X_train = X(1:N_train, :, :);
Y_train = Y(1:N_train, :, :);
clear X
clear Y

% form into complex data
if strcmp(inputType,'multi')
    X_train = reshape(X_train,size(X_train,1),N_elements*2,length(k_analyze));
    Y_train = reshape(Y_train,size(Y_train,1),N_elements*2,length(k_analyze));
    X_train = X_train(:, 1:N_elements, :) + 1j*X_train(:, N_elements+1:2*N_elements, :);
    Y_train = Y_train(:, 1:N_elements, :) + 1j*Y_train(:, N_elements+1:2*N_elements, :);
    X_train = reshape(X_train,size(X_train,1),N_elements*length(k_analyze));
    Y_train = reshape(Y_train,size(Y_train,1),N_elements*length(k_analyze));
else
    X_train = X_train(:, 1:N_elements, :) + 1j*X_train(:, N_elements+1:2*N_elements, :);
    Y_train = Y_train(:, 1:N_elements, :) + 1j*Y_train(:, N_elements+1:2*N_elements, :);
end

% save the data
if strcmp(inputType,'multi')
    kidx = round(median(k_analyze));
else
    kidx = k_analyze;
end
cnt=0;

% delete pre-existing h5 to avoid write error
if exist(fullfileOUT,'file'),delete(fullfileOUT);end

for k = kidx
    cnt=cnt+1;
    % set aperture data for specified frequency
    X_k = X_train(:, :, cnt);
    Y_k = Y_train(:, :, cnt);

    % create h5 file
    h5create(fullfileOUT, ['/' num2str(k-1) '/X/real'], size(X_k.'))
    h5create(fullfileOUT, ['/' num2str(k-1) '/Y/real'], size(Y_k.'))
    h5create(fullfileOUT, ['/' num2str(k-1) '/X/imag'], size(X_k.'))
    h5create(fullfileOUT, ['/' num2str(k-1) '/Y/imag'], size(Y_k.'))
    
    % write to h5 file
    h5write(fullfileOUT, ['/' num2str(k-1) '/X/real'], real(X_k.'))
    h5write(fullfileOUT, ['/' num2str(k-1) '/Y/real'], real(Y_k.'))
    h5write(fullfileOUT, ['/' num2str(k-1) '/X/imag'], imag(X_k.'))    
    h5write(fullfileOUT, ['/' num2str(k-1) '/Y/imag'], imag(Y_k.'))
    
end
%}
