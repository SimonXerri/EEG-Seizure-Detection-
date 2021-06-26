function f = extractFeature(signal)
f = [];
for i = 1:length(signal(1,:))
    s = signal(:,i);
    f1 = wave(s);
    f = [f f1];    
end