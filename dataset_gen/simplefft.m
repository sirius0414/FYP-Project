function varargout  = simplefft(tgtData,fftsize,presum,windowType,binsToKeep, displayTitle, tgtInfo)
            % Purpose:
            %   function to perform fft on timeseries raw complex samples from a single range and angle bin
            %   The fft legnth and presum in pulses is set. Waterfall plot
            %   of power spectrum in dB of time vs Doppler generated
            %
            %
            % Arguments:
            % tgtData This could be one of three types
            %         (1) Filename for data of type TgtGamekeeperRaw or array
            %         (2) data of class TgtGamekeeperRaw
            %         (3) Array of NxM data where N is the number of frames and M number of pulses in one frame
            %      if NxMx2 then a adjacent frame is added per frame
            % fftsize (optional), length of fft window
            % presum (optional), number of pulses to presum
            % windowType (optional), string for the fft weighting type
            % binsToKeep (optional),  The number of bins each side of Doppler 0 to keep.
            % displayTitle (optional), Title for the plot
            % tgtInfo (optional) , N x22 matrix with tgtInfo. One entry per  frame with 22 cloumns per frame. This is ignored if the input is of type TgtGamekeeperRaw
            % % Columns are  yyyy mm dd HH MM SS setno, timestep LatLongHgtPos, xyzPos, range, cosAz, cosEl, rangeGateIndex, BeamCol, BeamRow, DopplerBin, RadialSpeed
            % Outputs
            %  simplefft(...) returns no output
            %  fftData = simplefft(...) returns the fft of the timeseries data
            addpath F:\data\Code
            % Constants
            TGTINFONUMCOLOUMNS = 22;

            nOutputs = nargout;
            % Constants
            EXPANDPLOTDISPLAY = true ; % If true add more info to plot title
            displayTitleMore = '';
            argInNames = {'binsToKeep','displayTitle'}; % Used with GkLib.labelPlot
            
            
            % Initialise fft parameters
            
            % Gamekeeper system parameters - TODO should oobtain from gkParameter Class
            % object
            parm.prf = 7353;
            parm.wavelength = 0.2379;
            
            
            framesize = 2048; %TODO make this derived from input data second dimension.
            flagRepeatedPulse = false;
            
            % Option to read data from file. Renames the input var as tgtData.
            
            if ischar(tgtData)
                dataIn = load(tgtData);
                varIn = fields(dataIn);
                varIn = varIn{1}; % This assumes that the data has a single variable
                tgtData = dataIn.(varIn);
            end
            
            % Handle different dataTypes
            % TgtGamekeeperRaw has both rawData and truth
            % If just a matrix of rawData then use tgtInfo array as truth if passed as
            % an input arg
            switch class(tgtData)
                case 'TgtGamekeeperRaw'
                    data = tgtData.rawData;
                    tgtInfo = tgtData.truth;
                case 'double'
                    data = tgtData;
                    if nargin < 7
                        tgtInfo = [];
                    elseif ischar(tgtInfo)
                        if exist(tgtInfo,'file')
                            tgtInfo = csvread(tgtInfo,1,0);                            
                        else
                            warning('Unable to read targetInfo file %s. Skip plotting TargetInfo',tgtInfo)
                            tgtInfo = [];
                        end
                    end
            end
            
            if ~isempty(tgtInfo) && size(tgtInfo,2)~=TGTINFONUMCOLOUMNS
                warning('tgtInfo has incorrect size. Expected %d columns',TGTINFONUMCOLOUMNS)
                tgtInfo = [];
                
            end
            fftsizeIn = size(data,2);
            
            if ndims(data) ==3
                dataOrig = data;
                flagRepeatedPulse = true;
                data = data(:,1:framesize,1);
            end
            if nargin<2 || isempty(fftsize)
                fftsize = size(data,2);
            end
            if nargin<3|| isempty(presum)
                presum=1;
            end
            if nargin<4 || isempty(windowType)
                %     windowType = 'Hamming';
                windowType = 'BlackmanHarris';
            end
            
            if nargin<5 || isempty(binsToKeep)
                binsToKeep = 300;
            end
            binsToKeep = min(fftsize/2 - 1, binsToKeep);
            
            if nargin<6 || isempty(displayTitle)
                displayTitle = '';
            end
            
            dopplerBinspacing = 2* fftsize /(parm.wavelength*parm.prf/presum);
            
            parm.fftsize = fftsize;
            parm.windowType = windowType;
            
            binOrder=[fftsize-binsToKeep+1:fftsize 1:binsToKeep+1];
            
            if ~flagRepeatedPulse || fftsize<=framesize
                
                numFramesIn = size(data,1);
                numfftsizeIn = size(data,2);
                numSamplesIn = numfftsizeIn * numFramesIn;
                numFramesOut = floor(numSamplesIn/(fftsize*presum));
                
                % Reshape dats based on fftlength and presum
                dataTime = reshape(data',numSamplesIn,1);
                dataTime = dataTime(1:numFramesOut*fftsize*presum);
                dataTimePresum = sum(reshape(dataTime,presum,numFramesOut*fftsize)',2);
                dataReshaped = reshape(dataTimePresum,fftsize,numFramesOut)';
                
                % weight data and perform the fft
                fftwindow_rep = GkLib.buildWindow(parm,numFramesOut);
                windowed_data = dataReshaped.*fftwindow_rep;
                
                % TODO add the axis labels
            else
                if presum ~=1
                    warning('Presum not implemented for this option so resetting presum to 1')
                    presum = 1;
                end
                numFramesOut = size(dataOrig,1);
                numpulsesToCombine = fftsize/framesize;
                data = dataOrig(:,:,1:numpulsesToCombine);
                dataCollatedPulses = reshape(data,size(data,1),size(data,2)*numpulsesToCombine);
                dataCollatedPulses = dataCollatedPulses(:,2:framesize*numpulsesToCombine+1);
                
                % weight data and perform the fft
                fftwindow_rep = GkLib.buildWindow(parm,numFramesOut);
                windowed_data = dataCollatedPulses.*fftwindow_rep;
            end
            
            fftData = fft(windowed_data,[],2);
            fftData = fftData(:,binOrder);
            figure;waterfall(db(fftData));
            
            % Plot Axis labelling
            bin0 = binsToKeep+1;
            
            % displayTitle = regexprep(displayTitle,'\\','\\\'); % This fix is not required
            if EXPANDPLOTDISPLAY
                displayTitleMore = sprintf(' - %d %s fft',fftsize,windowType);
            end
            argIn.binsToKeep = binsToKeep;
            argIn.displayTitle = sprintf('Time-Doppler dB Spectrogram Plot%s\n%s',displayTitleMore,displayTitle);
            GkLib.labelPlot('spectrogram',argIn);
            
            if ~isempty(tgtInfo)
                if fftsize ~= fftsizeIn
                    warning('Skipping Tgtinfo overlay as this is curretnly not supported for the case where  requested fftsize %d is different from the original fftsize %d\n',fftsize,fftsizeIn);
                    return
                end
                tgtDoppler = -tgtInfo(:,22)*dopplerBinspacing+bin0;
                if presum >1
                    tgtDoppler = decimate(tgtDoppler,presum);
                end
                tgtTimestep = 1:length(tgtDoppler);
                
                %TODO check that tgtTimestep same length as data i.e same number of
                %frames
                hold on;
                plot(tgtDoppler,tgtTimestep,'o');
            end
            view(2);
            fh =  gcf;
            fh.WindowState = 'maximized';
            switch nOutputs
                case 0
                    % do nothing
                case 1
                    varargout{1} = fftData;
                otherwise
                    error('Too many output variables specified')
            end
end



             
      
