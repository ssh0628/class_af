
from scipy.signal import resample, butter, filtfilt
import numpy as np
import pandas as pd
import os
import re

def downsmapling(signal, original_rate=125, resampling_rate=32):
    resampling_length = int(len(signal) * resampling_rate / original_rate)
    return resample(signal, resampling_length)

def butterworth_filter(signal, lowcut=None, highcut=None, fs=64, order=4):
    nyq = 0.5 * fs
    if lowcut and highcut:
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
    elif lowcut:
        low = lowcut / nyq
        b, a = butter(order, low, btype='high')
    elif highcut:
        high = highcut / nyq
        b, a = butter(order, high, btype='low')
    else:
        return signal  # 필터 안 씀
    return filtfilt(b, a, signal)

def normalize_signal(signal, method='zscore'):
    if method == 'zscore':
        return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    elif method == 'minmax':
        min_val = np.min(signal)
        max_val = np.max(signal)
        return (signal - min_val) / (max_val - min_val + 1e-8)
    elif method == 'none':
        return signal
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def preprocess_signal(signal, fs=64, 
                      apply_filter=True, lowcut=0.5, highcut=8,
                      normalize=True, norm_method='zscore'):
    if apply_filter:
        signal = butterworth_filter(signal, lowcut=lowcut, highcut=highcut, fs=fs)
    if normalize:
        signal = normalize_signal(signal, method=norm_method)
    return signal

def split_signal(signal, segment_length):
    total_segment = len(signal) // segment_length
    return np.array(np.split(signal[:total_segment * segment_length], total_segment))

def labeling(filename):
    name = filename.lower()
    if 'non_af' in name:
        return 0
    elif 'af' in name:
        return 1
    else:
        raise ValueError(f"Labeling Error: {filename}")
    
def get_id(filename):
    match = re.search(r'(af|non_af)_\d{3}', filename.lower())
    if match:
        return match.group()
    else:
        raise ValueError(f"[Pasring Error] Invalid filename format: {filename}")

def ppg_signal(root_dir, save_root, segment_sec=25, original_rate=125, resampling_rate=32):
    segment_length = int(segment_sec * resampling_rate)
    os.makedirs(save_root, exist_ok=True)

    for dir_path, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.csv'):
                continue
            
            full_path = os.path.join(dir_path, filename)
            try:
                df = pd.read_csv(full_path)
                
                if 'PPG' not in df.columns:
                    print(f"There's no PPG column in {full_path}")
                    continue
                
                ppg = df['PPG'].values.astype(np.float32)
                
                # NaN 제거
                if np.isnan(ppg).any():
                    print(f"[Skip] NaN detected in {filename}, skipping.")
                    continue
                
                downsampled_ppg = downsmapling(ppg, original_rate, resampling_rate)
                
                preprocessed_ppg = preprocess_signal(downsampled_ppg, fs=resampling_rate,
                                                     apply_filter=True, lowcut=0.5, highcut=8,
                                                     normalize=True, norm_method='minmax')

                segments = split_signal(preprocessed_ppg, segment_length)

                label = labeling(filename)
                person_id = get_id(filename)

                person_dir = os.path.join(save_root, person_id)
                os.makedirs(person_dir, exist_ok=True)

                for idx, segment in enumerate(segments):
                    if np.isnan(segment).any():
                        print(f"[Skip] NaN in segment {idx} of {filename}, skipping.")
                        continue

                    save_name = f"segment_{idx:05d}.npz"
                    save_path = os.path.join(person_dir, save_name)
                    np.savez(save_path, ppg=segment, label=label)

            except Exception as e:
                print(f"[Error] {full_path}: {e}")


def save_to_npz(segments, save_path):
    os.makedirs(save_path, exist_ok=True)

    for idx, (segment, label) in enumerate(segments):
        filename = f"segment_{idx:05d}.npz"
        file_path = os.path.join(save_path, filename)
        np.savez(file_path, ppg=segment, label=label)

if __name__ == '__main__':
    # root_dir = r"C:\Users\cream\OneDrive\Desktop\Dataset\6973963\unzip_6973963"
    # save_dir = r"C:\Users\cream\OneDrive\Desktop\neuro\DB3"
    
    # signals = ppg_signal(root_dir, save_dir)
    # save_to_npz(signals, save_dir)
    # print(f"Save {len(signals)} segments in {save_dir}")
    
    """ import ast
    df = pd.read_csv("segment_00001.csv")
    label = df["label"].values[0]
    ppg = ast.literal_eval(df["ppg"].values[0]) """  # 문자열 리스트 → 실제 리스트로 변환