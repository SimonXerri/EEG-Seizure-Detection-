# EEG-Seizure-Detection
Researched and developed a seizure detection system using **MATLAB** and **Python** and investigated the effects of using classifier stacking for model accuracy enhancements. 
The [CHB-MIT scalp EEG database](https://physionet.org/content/chbmit/1.0.0/) was used for training and testing of the models.

Discrete Wavelet Transform was used to decompose the signal into 5-subband signals.
The following features were extracted from the subband signals:
* Standard Deviation
* Mean Absolute Deviation
* Root Mean Square
* Interquartile Range
* Max and Min

The classifiers used included: 
* Support Vector Machine
* Naive Bayes
* K-Nearest Neighbours
* Random Forest
* Multi-Layer Perceptron Neural Network
* Extreme Learning Machine

**Achieved model Sensitivity of 95%, Specificity of 97%, and Accuracy of 96%.**
