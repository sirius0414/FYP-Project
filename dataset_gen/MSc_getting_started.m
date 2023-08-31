clc;
clear;
%{

Welcome to getting_started.m

This script examples:
A) extracting, 
B) viewing, and 
C) analysing 
the data that has been shared with you (found in <basepath>\MSc2023\Data\...)

Feel free to use/copy any code in this script to help get yourself up and
running with our datasets.
At the same time, do go beyond the techniques used in this script in your
efforts to classify these spectograms


We will begin with the basics of extracting the data, 
after setting the path were our data is contained.
%}

basepath = 'F:\data'; % or might be 'D:\'



%% A) Extracting the data
%
% we will use 'fullfile' to join paths and folders together in a os
% independant way

real_data_path = fullfile(basepath, 'Data', 'Bird-Drone_Synthetic');
% ^ this is the folder that contains all the real drone and bird data

% There are a number of ways to get the files within a folder, and you can
% use any way you like. This is my favourite way:

real_data_list = dir(real_data_path);
real_data_cell = fullfile(real_data_path,{real_data_list(3:end).name});
disp(['There are ' num2str(length(real_data_cell)) ' files inside the folder: ' real_data_path])


% If the path has been set up properly, you will have a cell: https://uk.mathworks.com/help/matlab/ref/cell.html
% real_data_cell, that has a single data path for each index
disp(real_data_cell{50})

% to get the data from a path like this, we can do
% addpath(fullfile(basepath, 'Code'))
tgtData = load(real_data_cell{50}).tgtData;

% if you are getting the warning:
% % % Warning: Variable 'tgtData' originally saved as a TgtGamekeeperRaw cannot be
% % % instantiated as an object and will be read in as a uint32. 
% % % > In MSc_getting_started (line 43) 
% then make sure that the script, TgtGamekeeperRaw.m is on the MatLab path 
% (this script is found in the \MSc2023\Code folder, use addpath \.)



tgtData = load('F:\data\Data\Bird-Drone_Synthetic\Drone-261.mat').tgtData   % <--- This is the most simple form of the data inside MatLab

% You should see an object with 6 properties. You will be primarily
% interested in the 'rawData' one.

disp(['Size of the raw data in this sample is: ' num2str(size(tgtData.rawData,1)) 'x' num2str(size(tgtData.rawData,2))])
% >> Size of the raw data in this sample is: 622x2048


% Otherwise, the fastest way to load this data is to drag it into the
% Command Window, but this only allows you to look at one at a time.

% To look at the whole dataset and not just one example, you will need to
% loop over the paths inside real_data_cell one-at-a-time and get results.



%% B) Viewing the data
%
plt = @(fftData) imagesc(mag2db(abs(fftData)));
% As discussed with Dr. Antoniou, the spectogram is a key way to observe
% the raw data collected by a radar.
% The data shared with you is raw, timeseries, baseband I/Q data.
% It is in the time domain, so to view the spectrogram we will need to
% Fourier Transform it to the Frequency domain.
% We have provided the script, simplefft, that will allow you to do this
% quite easily. (again, this needs to be on the MatLab path like the other)
%
% And we will now demonstrate:
b=simplefft(tgtData,[],[],[],300); % 600=num Doppler bins either side of 0Hz
close
figure('Name','My First Spectogram Figure')
plt(b') % Here plt is an anonymous function that I defined above
colorbar
colormap('pink') % I find pink the easiest to see, you don't have to agree!



%% C) Analysing the data 
%
% In this case, 'b' is our spectral data. Feel free to rename this.
[nTimesteps, nDopplerBins] = size(b);
% This is how we will be interacting with the spectogram 
% Will now produce three figures exampling how the data is
figure('Name','A spectrum: complex')
title('Complex Timeseries Samples')
plot(b(100,:))
figure('Name','A spectrum: abs')
title('Absolute Spectrum in dB')
plot(abs(b(100,:)))
figure('Name','*A spectrum: abs, in dB*')
title('Absolute Spectrum in dB')
plot(mag2db(abs(b(100,:))))

% Where the final one is a spectrum of the same form that was found in the
% spectrogram image.

% As a final example, lets measure and print the SNR of the spectrum 
% we just created.
a_spectrum = mag2db(abs(b(100,:)));
peak_power= max(a_spectrum([1:596 604:1200])); % we ignore central few clutter bins 
noise_power_sample = mean(a_spectrum([1:60 1140:1200]));
SNR = peak_power - noise_power_sample;
disp(['Peak SNR of this spectrum in dB is: ' num2str(SNR)])

%% Good luck and happy classifying!
% Any questions, I am at dxw636@student.bham.ac.uk
% I have holiday 2nd-9th August, where responses will be slower.
%                                                                        //
