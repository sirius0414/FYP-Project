classdef TgtGamekeeperRaw <handle
    % TgtGamekeeperRaw class is raw Gamekeeper data for a specified target
    % The time series data is extracted from GamekeeperRaw rowColRT data
    % after it has been beamformed. Data is extracted from the resolution
    % cell(s) centered on position  given by TgtInfo CLass

    properties (Constant)
        TypeVersion = 1;
        % 1 = UoB data type with one resolution cell per pulse centred on target position
    end
    properties
        
        truth               % M x K matrix target information data listed for M frames where K = 22
        sourceRunFolderName % Gamekeeper source data run folder name (e.g CDR0-*)
        rawData             % raw data stored as MxN matrix where M is frames and N is pulsesNo
        frameTimes          % utc time of the frames as a M x1 array
        tgtLabel            % target label
        
    end
    
   methods
        % Constructor - Initialises the LabelledPlotC
        function obj = TgtGamekeeperRaw()  
            % Usage:
            %   TgtGamekeeperRaw(),  
            % Returns:
            %   obj - TgtGamekeeperRaw object
                
            obj.truth = [];
            obj.tgtLabel = '';
            obj.rawData = [];
            
        end  %Constructor
        
   end
    
   %TgtGamekeeperRaw CLASS implementation
    methods 

        function v = typeVersion(obj)
            % Purpose:
            %     produce type version for the concrete type
            % Returns:
            %     type version as uint32
            v = obj.TypeVersion;
        end
        
        function setTgtInfo(obj, tgtInfo, tgtLabel)
            % Purpose
            %     Initialise the tgtInfo properties of the class
            obj.truth = tgtInfo;
            obj.frameTimes = datenum(obj.truth(:,1:6));
            if nargin==3
                obj.tgtLabel = tgtLabel;
            end
        end
        
        function setSourceFolder(obj, sourceRunFolderName)
            % Purpose
            %     Add sourceRunFolderName to Class property
            obj.sourceRunFolderName = sourceRunFolderName;
        end
    end
end

