import pickle
import numpy as np
import time
import os
import pywt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# constants
base_dir = os.path.join('..', 'data', 'numpy_data')
sampling_rate = 256  # Hz
duration = 10  # seconds
num_samples = duration * sampling_rate

# load data
def get_data(seizure: bool):
    if seizure == True:
        with open('clusters_seizure.pkl', 'rb') as file:
            clusters = pickle.load(file)
    else:
        with open('clusters_nonseizure.pkl', 'rb') as file:
            clusters = pickle.load(file)
    return clusters

# wavelet decomposition

def wavelet_decompose(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs

def extract_wavelet_coeff_stats_features(coeffs):
    features = []
    for coeff in coeffs:
        features.append(np.mean(coeff))
        features.append(np.std(coeff))
    return features

def wavelet_coeff_feature_extractor(x):
    coeff = np.array(x)
    energy = np.sum(coeff ** 2)
    entropy = -np.sum((coeff ** 2) * np.log(coeff ** 2 + 1e-12))
    std_dev = np.std(coeff)
    variance = np.var(coeff)
    mean_val = np.mean(coeff)
    median_val = np.median(coeff)
    return [energy, entropy, std_dev, variance, mean_val, median_val]

def wavelet_coeff_feature_list(data):
    # data must be a 2D numpy array where rows are channels and columns are data points
    all_features = []
    for channel_data in data:
        coeffs = wavelet_decompose(channel_data)
        features = extract_wavelet_coeff_stats_features(coeffs)
        all_features.append(features)
    return all_features

def extract_wavelet_features(segmentations):
    features = []
    p = 0
    for patient in segmentations:
        filename = 'preprocessed_' + str(p) + '.npy'
        full_file_path = os.path.join(base_dir, filename)
        try:
            data = np.load(full_file_path)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            p += 1
            continue

        for subsegment in patient:
            start, end = subsegment
            if end * 256 > data.shape[1]:  # Check if the end index is within the bounds
                print(f"Index out of bounds for {filename}: start={start * 256}, end={end * 256}")
                continue

            try:
                features = wavelet_coeff_feature_list(data[:, int(start * 256): int(end * 256)])
                subfeatures = []
                for i in features:
                    subfeatures.append(extract_wavelet_features(i))
                features.append(subfeatures)
            except Exception as e:
                print(f"Error processing data in {filename} from {start * 256} to {end * 256}: {e}")
    
    p += 1
    print(p)
    return features

# prepare features to tabular format
def transform_features(data):
    data = np.array(data)
    data_flat = []
    for i in range(len(data)):
        data_flat.append(data[i].flatten())
    data_flat_T = np.array(data_flat).T
    return data_flat
    return data_flat_T

def get_df(s_features, ns_features):
    df_s = pd.DataFrame(transform_features(s_features))
    df_ns = pd.DataFrame(transform_features(ns_features))
    df_s['label'] = 1
    df_ns['label'] = 0
    df = pd.concat([df_s, df_ns], ignore_index=True)
    df.to_csv('wavelet_features.csv', index=False)
    return df

# prepare features for modelling
def prepare_features(df):
    X = df.drop(columns=['label'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test

# make model, train and evaluate
def model_train_eval(model, X_train, y_train, X_test, y_test):
    fitted_model = model.fit(X_train, y_train)
    y_pred = fitted_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# pipeline for feature extraction
def feature_pipeline():
    data_s = get_data(seizure=True)
    data_ns = get_data(seizure=False)

    all_features_s = extract_wavelet_features(data_s)
    all_features_ns = extract_wavelet_features(data_ns)

    all_features_s = transform_features(all_features_s)
    all_features_ns = transform_features(all_features_ns)
    
    df = get_df(all_features_s, all_features_ns)
    X_train, X_test, y_train, y_test = prepare_features(df)

    return X_train, X_test, y_train, y_test