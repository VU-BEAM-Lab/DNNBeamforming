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

% clear workspace
clear all; close all;

% set up params
fpath = 'example_phantom_10mm_70mm/';
crange = [0.058,0.078];
inputType = 'single';
k_analyze = 9;
N_window = 16;
target_idx = 1:10;

% for each target
for k=1:length(target_idx)
    % specify full file paths to input and output data
    fullfileIN = fullfile(fpath,['chandat',num2str(target_idx(k)),'.mat']);
    fullfileOUT = strrep(fullfileIN,'.mat','.h5');
    % make training data
    createTrainingData(fullfileIN,fullfileOUT,crange,inputType,...
                    k_analyze,N_window)
end