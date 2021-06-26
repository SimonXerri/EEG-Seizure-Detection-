function y = filterEEG(x)
persistent Hd;

if isempty(Hd)
    
    %The following code was used to design the filter coefficients:
    
%     N     = 2;    % Order
%     F3dB1 = 0.5;  % First
%     F3dB2 = 25;   % Second
%     Fs    = 256;  % Sampling Frequency
%     
%     h = fdesign.bandpass('n,f3db1,f3db2', N, F3dB1, F3dB2, Fs);
%     
%     Hd = design(h, 'butter', ...
%         'SystemObject', true);
%     
%     Hd.Structure
%     Hd.SOSMatrixSource
%     Hd.SOSMatrix
%     Hd.ScaleValues
%     Hd.InitialConditions
%     Hd.OptimizeUnityScaleValues
    
%     Hd = dsp.BiquadFilter( ...
%         'Structure', 'Direct form II', ...
%         'SOSMatrix', [1 0 -1 1 -1.30149453646959 0.310059809032142], ...
%         'ScaleValues', [0.344970095483929; 1]);


    Hd = dsp.BiquadFilter( ...
        'Structure', 'Direct form II', ...
        'SOSMatrixSource', 'Property', ...
        'SOSMatrix', [1.0000 0 -1.0000 1.0000 -1.5207 0.5266], ...
        'ScaleValues', [0.2367; 1.0000], ...
        'InitialConditions', 0, ...
        'OptimizeUnityScaleValues', true);

end

s = double(x);
for i = 1:23
    y(:,i) = step(Hd,s(:,i));
end

