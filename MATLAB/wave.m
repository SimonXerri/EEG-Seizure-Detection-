function f = wave(signal)

% 5 level wavelet decomposition
[c,l] = wavedec(signal,5,'db4');

% approximation coef
ap5 = appcoef(c,l,'db4');

% detail coef
[cd1,cd2,cd3,cd4,cd5] = detcoef(c,l,[1 2 3 4 5]);

% std
f1 = std(cd1);
f2 = std(cd2);
f3 = std(cd3);
f4 = std(cd4);
f5 = std(cd5);
f6 = std(ap5);

% Mean absolute deviation
f7 = mad(cd1);
f8 = mad(cd2);
f9 = mad(cd3);
f10 = mad(cd4);
f11 = mad(cd5);
f12 = mad(ap5);

% Root Mean Square
f13 = rms(cd1);
f14 = rms(cd2);
f15 = rms(cd3);
f16 = rms(cd4);
f17 = rms(cd5);
f18 = rms(ap5);

% Min
f19 = min(cd1);
f20 = min(cd2);
f21 = min(cd3);
f22 = min(cd4);
f23 = min(cd5);
f24 = min(ap5);

% Max
f25 = max(cd1);
f26 = max(cd2);
f27 = max(cd3);
f28 = max(cd4);
f29 = max(cd5);
f30 = max(ap5);

% IQR
f31 = iqr(cd1);
f32 = iqr(cd2);
f33 = iqr(cd3);
f34 = iqr(cd4);
f35 = iqr(cd5);
f36 = iqr(ap5);


f = [f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 f27 f28 f29 f29 f30 f31 f32 f33 f34 f35 f36];