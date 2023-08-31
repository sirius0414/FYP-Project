classdef GkLib 
    % Class to contain set off static method used with radar data
    % processing and displau

    
    methods   ( Static = true )
        
        function labelPlot(plotType,argIn)
            % Purpose:
            %  Provide a generic routine for labelling plots of different
            %  types
            %  Argument:
            %    plotType, string controlling the plot configurations
            %    inputArg, struct to pass input arguments
            
            switch plotType
                case {'spectrogram' 'rangeDoppler'}
                    requiredFields = {'binsToKeep','displayTitle'};
                    if ~min(ismember(requiredFields,fields(argIn)))
                        warning('Missing  field %s from "argIn" for labelPlot\n Skiping Plot Labelling',requiredFields{~ismember(requiredFields,fields(argIn))})
                        return
                    end
                    GkLib.labelPlotDoppler(plotType,argIn.binsToKeep,argIn.displayTitle)
                    
                otherwise
                    warning('Skiping plot labelling as plotType %s not recognised',plotType)
            end    
        end
        
        function labelPlotDoppler(plotType,binsToKeep,displayTitle)
            % Purpose:
            %  Routine to label spectrogram waterfalls and their X-Axis for
            %  Doppler ticks
            %  types
            %  Argument:
            %    plotType, string controlling the plot configurations            
            %    binsToKeep, half width of Doppler range
            %    displayTitle, string for plot title            
            bin0 = binsToKeep+1;
            if binsToKeep < 500
                binstep = 100;
            else
                binstep = 200;
            end
            xTickVals = [(bin0-1000):binstep:bin0 (bin0+100):binstep:(bin0+1000)];
            xTickLabels = [-1000:binstep:0,100:binstep:1000];
            
            title(displayTitle,'Interpreter','none') ;
            set(gca,'XTick', xTickVals, 'XTickLabel', xTickLabels );
            xlabel('Doppler Bin')
            switch plotType
                case 'spectrogram' 
                    ylabel('Time Steps'), zlabel('dB')
                case 'rangeDoppler'
                    ylabel('Range Bin');
            end
            
            % Max the axis font larger but keep title small
            set(gca,'FontSize',25);
            ax = gca;
            ax.TitleFontSizeMultiplier = 0.75;
        end
        
        function fftwindow_rep = buildWindow(parm, replicateDimensions)
            % Generates the chosen windowing function to be applied to data taking into
            % the replicate dimensions
            
            % Arguments:
            %   parm  is input structure with following fields
            %    parm.fftsize:  the length of the window.
            %    parm.windowType window type
            % Properties used as output:
            %   fftwindow_rep, the fft window replicated in
            %   replicateDimensions with fft window in the last dimension
            
            
            if nargin<2
                replicateDimensions = [];
            end
            
            
            % Check if required fields in parm
            requiredFields = {'windowType','fftsize'};
            if ~min(ismember(requiredFields,fields(parm)))
                error('Missing data field %s from "parm" for buildWindow\n',requiredFields{~ismember(requiredFields,fields(parm))})
            end
            
            
            % Calculate the window in one dimension
            switch parm.windowType
                case 'Rectangular'
                    fftwindow = ones(parm.fftsize, 1);
                case 'Hamming'
                    % We compute it directly to avoid need for signal processing toolbox
                    fftwindow = 0.54 - 0.46 * cos(2 * pi * (0:parm.fftsize-1)' / (parm.fftsize-1));
                case 'BlackmanHarris'
                    % This requires signal processing toolbox, but there is an alternative in signalProcessingReplicants.
                    fftwindow = blackmanharris(parm.fftsize);
                case 'Chebyshev'
                    fftwindow = chebwin(parm.fftsize);
                otherwise
                    error ('Unrecognised windowType %s', parm.windowType);
            end
            
            % Note the fft window is scale down so that the amplitude gain of the FFT is unity.
            fftwindow_gain = sum(fftwindow);
            fftwindow = fftwindow / fftwindow_gain;
            
            switch length(replicateDimensions)
                case 0
                    fftwindow_rep = fftwindow;
                case 1
                    fftwindow_rep = zeros(1,parm.fftsize);
                    fftwindow_rep(1,:) = fftwindow;
                case 3
                    fftwindow_rep = zeros(1, 1, 1, parm.fftsize);
                    fftwindow_rep(1,1, 1, :) = fftwindow;
                    
                otherwise
                    error('fft window build not supported for dimensions %d',length(replicateDimensions))
            end
            fftwindow_rep = repmat(fftwindow_rep,[replicateDimensions, 1]);
            
        end
    end
end

