# ecg-noise-detector

This package helps detect and manage noise in ECGs

## 📚 Original publication

Yet to be published

## 🔧 Instalation

This package is available via pip
```
pip install ecg-noise-detector
```

## 💻 Get started

It is important that raw ECGs (i.e not bandpass filtered) are used when using `is_noisy` and `plot_ecg`

```python
from ecg_noise_detector import noiseDetector

# Generate a noisy ECG
ecg = noiseDetector.get_example_ecg('noisy')

# Plot the ecg with green highlights on where clean signal is present
noiseDetector.plot_ecg(ecg)

# Classify the ecg
print(noiseDetector.is_noisy(ecg))

```

## 📒 Quick Specification

```python
get_example_ecg(ecgType)
'''
RETURNS 
Numpy array of 30s raw ECG of specified type

INPUTS
@ ecgType - ['clean' | 'noisy'], specifies which ecg to generate
'''

plot_ecg(ecg, fs=500, highlights=True, show=True)
'''
RETURNS 
pyplot figure of filtered (and highlighted) ecg

INPUTS
@ ecg - raw ecg
@ fs - sampling frequency
@ highlights - show highlights (green when segment is clean, grey when noisy)
@ show - display figure when function is executed
'''

is_noisy(ecg, fs=500)
'''
RETURNS 
boolean (True if ecg is noisy, False if not)

INPUTS
@ ecg - raw ecg
@ fs - sampling frequency
'''

```



