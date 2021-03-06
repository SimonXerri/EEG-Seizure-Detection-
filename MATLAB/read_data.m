function sampleTable = read_data(recordName)

% add toolbox to path
addpath([pwd '/mcode']);
% record path from physionet.org
recordPath = ['chbmit/1.0.0/' recordName];
% read annotation
ann = rdann(recordPath, 'seizures');
% beginning and end of seizures
annStart = [];
annEnd = [];
for i = 1:2:length(ann)-1
    annStart = [annStart; ann(i)];
    annEnd = [annEnd; ann(i+1)];
end
% calculate total length of seizures
lenghtSeizures = 0;
for i = 1:length(annStart)
    lenghtSeizures = lenghtSeizures + annEnd(i)-annStart(i)+1;
end
% common channels
commonChannels = ["FP1-F7" "F7-T7" "T7-P7" "P7-O1" "FP1-F3" "F3-C3"...
    "C3-P3" "P3-O1" "FZ-CZ" "CZ-PZ" "FP2-F4" "F4-C4" "C4-P4" "P4-O2"...
    "FP2-F8" "F8-T8" "T8-P8" "P8-O2" "P7-T7" "T7-FT9" "FT9-FT10" "FT10-T8"];


%% read EEG segments
% read description
info = wfdbdesc(recordPath);
% extract channel names
for i = 1:length(info)
    channels(i) = string(info(i).Description);
end
% common channel index
idx = [];
for i = 1:length(commonChannels)
    idx = [idx find(commonChannels(i) == channels)];
end
% sort index
idx = sort(idx);
% read seizure free segment
if lenghtSeizures > annStart(1)
    normal = rdsamp(recordPath, idx, annEnd(1)+lenghtSeizures, annEnd(1)+1);
    if annEnd(1)+lenghtSeizures > annStart(2)
        fprintf('Error2\n\n\n')
    end
else
    normal = rdsamp(recordPath, idx, lenghtSeizures);
end
    
% read seizure segments
seizure = [];
for i = 1:length(annStart)
    seizure = [seizure; rdsamp(recordPath, idx, annEnd(i), annStart(i))];
end


%% filter signals
normal = filterEEG(normal);
seizure = filterEEG(seizure);
% create 2 seconds segment
sampleTable = table();
% segment lenght
segLen = 512;
% number of segments
numSegment = floor(length(normal)/segLen);
for i = 1:numSegment
    % normal signal
    class = categorical("Normal");
    signal = normal(segLen*(i-1)+1:segLen*i,:);
    feature = extractFeature(signal);
    temp = table(feature, class);
    sampleTable = [sampleTable; temp];
    % seizure
    class = categorical("Seizure");
    signal = seizure(segLen*(i-1)+1:segLen*i,:);
    feature = extractFeature(signal);
    temp = table(feature, class);
    sampleTable = [sampleTable; temp];
end
    