clc;
clear;
FRACLENGTH = 20; %20 timestep of each example
basepath = 'F:\data';
savepath = fullfile(basepath,'Saved');
plt = @(fftData) imagesc(mag2db(abs(fftData)));


%% Extracting the data
real_data_path = fullfile(basepath, 'Data', 'Bird-Drone_Synthetic');
real_data_list = dir(real_data_path);
real_data_cell = fullfile(real_data_path,{real_data_list(3:end).name});
disp(['There are ' num2str(length(real_data_cell)) ' files inside the folder: ' real_data_path])

dataset_len = length(real_data_cell);
for m = 1:dataset_len
    tgtData = load(real_data_cell{m}).tgtData;
    if size(tgtData.rawData,1) > FRACLENGTH
        num_data_frac = floor(size(tgtData.rawData,1) / FRACLENGTH);
        for n = 1:num_data_frac
            rawData_frac = tgtData.rawData(FRACLENGTH*(n-1)+1:FRACLENGTH*n,:);
            disp(['The data fraction ' num2str(n) ' of the example ' num2str(m) ' has a size of: ' num2str(size(rawData_frac,1)) 'x' num2str(size(rawData_frac,2))])
            fftData_frac = simplefft(rawData_frac,[],[],[],600);
            close
%             figure('Name','My First Spectogram Figure')
%             plt(fftData_frac') % Here plt is an anonymous function that I defined above
%             colorbar
%             colormap('pink')
            data_name = [tgtData.tgtLabel , '_' ,num2str(m) , '-' , num2str(n), '.mat'];
            label = tgtData.tgtLabel;
            save(['F:\data\Data\Synthetic\', data_name],"label","fftData_frac",'-v6');
        end
    else
        disp('the length of this example is too short')
    end
end
