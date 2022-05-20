import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import os
from scipy.signal import butter,filtfilt
import math
import pandas as pd
from ecgdetectors import Detectors
from scipy.signal import welch
from scipy.integrate import simps
from scipy.stats import skew, kurtosis
from scipy import interpolate
from scipy import signal
from joblib import load

def get_example_ecg(ecgType = "clean"):
    if ecgType not in ['clean', 'noisy']:
        raise ValueError("invalid ecg type provided")

    file = "clean_ecg.csv" if ecgType == "clean" else "noisy_ecg.csv"
    return np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', file))

def plot_ecg(ecg, fs=500, highlights=True, show=True):
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(11.69, 8.27)
    
    x = np.linspace(0, 30, num=30*fs)
    filtered_ecg = _filter_ecg(ecg, fs)
    
    for i in range(3):
        pt1 = int(i*10*fs)
        pt2 = int((i+1)*10*fs)
        
        axs[i].set_xticks(np.arange(0+(i*10),10.01+(i*10),0.2))
        axs[i].set_yticks(np.arange(0,1.1,0.14))
        axs[i].xaxis.set_major_formatter(plt.NullFormatter())
        axs[i].yaxis.set_major_formatter(plt.NullFormatter())

        axs[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        axs[i].yaxis.set_minor_locator(AutoMinorLocator(5))
        
        axs[i].grid(which='major', linestyle='-', linewidth='0.5', color='black')
        axs[i].grid(which='minor', linestyle='-', linewidth='0.5', color='lightgray')
            
        axs[i].axis(xmin=0+(i*10),xmax=10+(i*10),ymin=-0.05,ymax=1.1)
        axs[i].plot(x[pt1:pt2], filtered_ecg[pt1:pt2], 'k')
    
    if highlights == True:
        segments = _process_ecg(ecg)
        model = load(os.path.join(os.path.dirname(__file__), 'data', 'model_20_05_21.joblib'))
        predictions = model.predict(segments)
        
        rolling = 0.5
        window = 5
        for i in range(len(predictions)):
            pt1 = i*rolling
            pt2 = pt1+window
            plt_n = 0 if pt1 <10 else 1 if pt1 < 20 else 2
            if predictions[i] == 0:
                axs[plt_n].axvspan(pt1, pt2, facecolor='lightgreen', alpha=1)
                if (pt2 > 10 and pt2 < 15) or (pt2 > 20 and pt2 < 25):
                    axs[plt_n+1].axvspan(pt1, pt2, facecolor='lightgreen', alpha=1)
    if show == True:
        plt.show()
    return fig

def is_noisy(ecg, fs=500):
    segments = _process_ecg(ecg)
    model = load(os.path.join(os.path.dirname(__file__), 'data', 'model_20_05_21.joblib'))
    predictions = model.predict(segments)

    if sum(predictions) < len(segments)//2:
        return False
    return True

def _process_ecg(ecg, fs=500):
    data = []
    raw_data = []
    sampling_freq = fs
    down_freq = 300
    window = 5
    rolling = 0.5
    filtered_ecg = _filter_ecg(ecg, fs)

    reached_end = False
    for i in range(0, len(ecg)+window*(sampling_freq), int(rolling*(sampling_freq))):
        if i+(window*(sampling_freq)) <= len(ecg):
            curr = filtered_ecg[i:i+window*(sampling_freq)]
            curr_raw = ecg[i:i+window*(sampling_freq)]
        else:
            reached_end = True
            curr = filtered_ecg[i:]
            curr = np.pad(curr, (0, window*sampling_freq-len(curr)), 'reflect')
            
            curr_raw = ecg[i:]
            curr_raw = np.pad(curr, (0, window*sampling_freq-len(curr)), 'reflect')
        y = np.zeros(3*len(curr))
        y_raw = np.zeros(3*len(curr))
        j = 0
        for i in range(0, len(curr)):
            y[j] = curr[i]
            y[j+1] = curr[i]
            y[j+2] = curr[i]
            
            y_raw[j] = curr_raw[i]
            y_raw[j+1] = curr_raw[i]
            y_raw[j+2] = curr_raw[i]
            j+=3
            
        data.append(y.reshape(-1, 5).mean(axis=1))
        raw_data.append(y_raw.reshape(-1, 5).mean(axis=1))
        
        if reached_end:
            break

    processed_ecg = pd.DataFrame(columns=['sSQI','kSQI','pSQI','basSQI','bSQI','rSQI'])
    count = 0
    for i in range(len(data)):
        processed_ecg.loc[count] = _get_SQIs(data[i],raw_data[i])
        count += 1
    
    sSQI_u = 0.593
    kSQI_u = 12.19

    sSQI_var = 4.87
    kSQI_var = 218.52


    processed_ecg['sSQI'] = processed_ecg['sSQI'].apply(lambda x: (x-sSQI_u) / np.sqrt(sSQI_var))
    processed_ecg['kSQI'] = processed_ecg['kSQI'].apply(lambda x: (x-kSQI_u) / np.sqrt(kSQI_var))

    return processed_ecg

def _butter_lowpass_filter(data, cutoff):
    T = 10         # Sample Period
    fs = 150      # sample rate, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2
    n = int(T * fs) # total number of samples

    normal_cutoff = cutoff / nyq
    
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def _length_transfrom(x, w):
    fs = 300
    tmp = []
    for i in range(w, len(x)):
        curr = 0
        for k in range(i-w+1, i):
            curr += np.sqrt((20/fs)+np.power(x[k]-x[k-1],2))
        tmp.append(curr)
    l = [tmp[0]]*w
    
    return l+tmp

def _threshold(x):
    u = np.mean(x)
    peaks = []
    fs = 300
    for i in range(len(x)):
        if (len(peaks) == 0 or i > peaks[-1]+(fs*0.18)) and x[i] > u:
            peaks.append(i)
    return peaks

def _wqrs(x):
    fs = 50
    y = _butter_lowpass_filter(x, 15)
    y = _length_transfrom(y, math.ceil(fs*.130))
    return _threshold(y)

def _get_SQIs(x, raw_x):
    down_freq = 300
    sSQI = skew(x, bias=False)
    kSQI = kurtosis(x, fisher=False, bias=False)
    
    def get_first_idx(x, a):
        for i in range(len(x)):
            if x[i] >= a:
                    return i

    def get_last_idx(x, a):
        for i in range(len(x)):
            if x[i] >= a:
                return i-1
                
    f, Pxx_den = welch(raw_x, fs=300)
    u1 = simps(Pxx_den[get_first_idx(f, 5):get_last_idx(f, 15)], x=f[get_first_idx(f, 5):get_last_idx(f, 15)])
    u2 = simps(Pxx_den[get_first_idx(f, 5):get_last_idx(f, 40)], x=f[get_first_idx(f, 5):get_last_idx(f, 40)])
    u3 = simps(Pxx_den[get_first_idx(f, 1):get_last_idx(f, 40)], x=f[get_first_idx(f, 1):get_last_idx(f, 40)])
    u4 = simps(Pxx_den[get_first_idx(f, 0):get_last_idx(f, 40)], x=f[get_first_idx(f, 0):get_last_idx(f, 40)])

    pSQI = (u1/u2)
    basSQI = (u3/u4)
            
    detectors = Detectors(down_freq)
    
    wqrs = _wqrs(x)
    eplimited = detectors.hamilton_detector(x)
    count = 0
    j = 0
    k = 0
    while j < len(wqrs) and k < len(eplimited):
        if wqrs[j] >= eplimited[k]-0.13*down_freq and wqrs[j] <= eplimited[k]+0.13*down_freq:
            count += 1
            j+=1
            k+=1
        elif wqrs[j] > eplimited[k]:
            k+=1
        else:
            j+=1

    bSQI = (count/len(wqrs))
    rSQI = (len(wqrs)/(len(wqrs)+len(eplimited)))
    
    return [sSQI, kSQI, pSQI, basSQI, bSQI, rSQI]

def _filter_ecg(x, fs):
    b, a = signal.butter(3, [0.005, 0.06], 'band')
    x = signal.filtfilt(b, a, x, padlen=150)
    x = (x - min(x)) / (max(x) - min(x))
    return x