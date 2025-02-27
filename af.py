import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt, iirnotch
import scipy.signal as signal
import numpy as np
import pywt

ds_path = r"C:\Users\Chao\Downloads\intracardiac-atrial-fibrillation-database-1.0.0\intracardiac-atrial-fibrillation-database-1.0.0"
ds_name = "iaf1_afw"
ds = wfdb.rdrecord(f"{ds_path}/{ds_name}")
# annotation = wfdb.rdann(f"{ds_path}/{ds_name}", 'qrs')
fs=ds.fs

n_samples = int(1 * fs)
data_3s = ds.p_signal[:n_samples]

#raw ecg signal 
ecg_raw = data_3s[:,0]

#noise reduction 
def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    ny = 0.5* fs
    low = lowcut / ny
    high = highcut / ny
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, freq=50, fs=1000, Q=30):
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, signal)

def highpass_filter(signal, cutoff=0.5, fs=1000, order=4):
    nyquist = 0.5 * fs
    high = cutoff / nyquist
    b, a = butter(order, high, btype='high')
    return filtfilt(b, a, signal)

def wavelet_denoising(signal, wavelet="db6", level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, np.std(c) * 0.5, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

ecg_band = bandpass_filter(ecg_raw, 20, 250, fs)
ecg_notch = notch_filter(ecg_band)
ecg_hp = highpass_filter(ecg_notch)
ecg_wv = wavelet_denoising(ecg_hp)

# qrs detection - pan tompkins
# calc diff
ecg_diff = np.diff(ecg_wv)
# square
ecg_squa = ecg_diff ** 2
# mv - smoothing
window_size = int(0.15 * fs)
kernel = np.ones(window_size) / window_size
ecg_mv = np.convolve(ecg_squa, kernel, mode='same')
# threshold
threshold = np.mean(ecg_mv) * 1.2
peaks, _ = signal.find_peaks(ecg_mv, height=threshold, distance=fs*0.6)
# plot
num_samples = len(ecg_raw)
time = np.arange(num_samples) / fs  # Time in seconds

plt.figure(figsize=(10,4))
# # plt.scatter(annotation.sample[start_idx:end_idx], [ds.p_signal[i, 1] for i in annotation.sample[start_idx:end_idx]], 
# #            color='red', label="QRS Peaks")
plt.plot(time, ecg_raw, label="Raw ECG", color='gray')
plt.plot(time, ecg_band, label="Bandpass ECG", color='r')
plt.plot(time, ecg_notch, label="Notch ECG", color='g')
plt.plot(time, ecg_hp, label="Highpass ECG", color='b')
plt.plot(time, ecg_wv, label="Wavelet ECG", color='m')

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.title("ECG Signal")

# plt.figure(figsize=(10, 4))
# plt.plot(time[:-1], ecg_mv, label="Wavelet ECG", color='m')

# plt.plot(time[:-1], ecg_mv, label="MV Signal", color='blue')
# plt.scatter(time[peaks], ecg_mv[peaks], color='black', label="QRS Peaks", marker='o')
# plt.title("QRS Detection")
# plt.legend()

plt.show()

# Print general info
print(f"Sampling Frequency: {ds.fs} Hz")
print(f"Signal Shape: {ds.p_signal.shape}")  # (n_samples, n_channels)
print(f"Signal Names: {ds.sig_name}")
print(f"II signal index: {ds.sig_name.index('aVF')}")
print(f"Units: {ds.units}")