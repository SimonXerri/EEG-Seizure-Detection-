clc; clear all; close all;

%% read and extract features
% read record names from text file
fileID = fopen('seizure_records.txt');
temp = textscan(fileID, '%s');
records = temp{1};

% create table containing features and class
dataTable = table();
for i = 1:length(records)
    fprintf('Record %d is being processed out of %d records\n',i,length(records));
    dataTable = [dataTable; read_data(records{i})];    
end
save("Extracted_Features.mat", "dataTable");

loadedTable = load('Extracted_Features.mat');
dataTable = struct2table(loadedTable);

writetable(dataTable.(1),'Extracted_Features.csv','Delimiter',',');




