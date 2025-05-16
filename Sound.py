import time
from collections import defaultdict
from enum import Enum

import librosa
from librosa.display import specshow
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy
import scipy.signal.windows as wd
from librosa.feature import spectral_centroid
import cmath
import soundfile as sf
import math
import pandas as pd
from pandas import plotting
from sympy import false
from torchaudio.functional import spectrogram

import Filters

from numba.core.itanium_mangler import mangle

from Filters import low_pass_filter, high_pass_filter


def phasor_output(magnitude, angle):
    return math.cos(angle) * magnitude, -math.sin(angle) * magnitude

def discrete_fourier_transforms(samples):
    omega_k = [2 * np.pi * (k/len(samples)) for k in range(len(samples))]

    def single_frequency(frequency_rad, samples):

        reals = [math.cos(frequency_rad * i) * samples[i] for i in range(len(samples))]
        imaginaries = [-math.sin(frequency_rad * i) * samples[i] for i in range(len(samples))]

        real_sum = np.sum(reals)
        imaginary_sum = np.sum(imaginaries)

        magnitude = ((real_sum ** 2 + imaginary_sum ** 2)**0.5) / len(samples)
        magnitude = np.abs(magnitude)
        return magnitude


    amplitudes = list(map(lambda x: single_frequency(x, samples), omega_k))

    return omega_k, amplitudes

def fast_fourier_transform(samples, remove_mirror = True, uncomplex = True):
    raw_fft = np.fft.fft(samples)
    if remove_mirror: raw_fft = raw_fft[:len(samples)//2]
    normalized_fft = raw_fft# / len(samples)
    non_complex_fft = np.abs(normalized_fft)
    fft = np.abs(raw_fft) if uncomplex else raw_fft
    return fft

def generate_windows(samples:list, window_size:int, overlap:float, n_fft=2048, shape = None):
    if shape is None: shape=np.arange(1, window_size + 1)
    windows = []
    for i in range(math.floor((len(samples)/window_size) / (1 - overlap))):
        first_slice = math.floor(max(0.0, i * window_size - window_size * i * overlap))
        second_slice = min(math.floor(first_slice + window_size), len(samples))
        window = samples[first_slice:second_slice]
        if len(window) < window_size:
            window = np.append(window, [0 for _ in range(window_size - len(window))])

        window = np.multiply(window, shape)

        window = np.append(window, [0 for _ in range(n_fft - window_size)])
        windows.append(window)

    return windows

def make_spectrogram(windows:list, keep_raw = False):

    spectrogram:np.float32 = np.array([windows[0][:len(windows[0])//2]] if not keep_raw else [windows[0]], dtype=np.float32)
    for window in windows:
        fft = fast_fourier_transform(window) if not keep_raw else fast_fourier_transform(window, False, False)
        spectrogram = np.append(spectrogram, [fft], axis=0)
    spectrogram = np.delete(spectrogram, [0], axis=0)

    return spectrogram

def map_sinus_frequency_to_hz(sinus_frequency:int, max_frequecy:int, sampling_rate:int):
    base = (sampling_rate * 0.5)/max_frequecy
    return base * sinus_frequency

def get_spectral_centroids(spectrogram, n_fft, sampling_rate):
    def calculate_centroid_for_spectrogram_window(spectral_window: list, n_fft, sampling_rate):
        numerator = 0
        denominator = 0
        for frequency in range(len(spectral_window)):
            magnitude = spectral_window[frequency]
            frequency = map_sinus_frequency_to_hz(frequency, n_fft, sampling_rate)

            numerator += magnitude * frequency
            denominator += magnitude

        return numerator / denominator if not np.isnan(numerator / denominator) else 0

    spectral_centroids = list(map(lambda x: calculate_centroid_for_spectrogram_window(x, n_fft, sampling_rate), spectrogram))
    return spectral_centroids

def get_spectral_spread(spectrogram, centroids, n_fft, sampling_rate):

    def spread_for_frame(frame, centroid): #Very similar to the standard deviation formula
        numerator = 0 #
        denominator = 0 #Total frequency strength in whole frame
        for frequency in range(len(frame)):
            magnitude = frame[frequency]
            frequency = map_sinus_frequency_to_hz(frequency, n_fft, sampling_rate)
            diff_sq = (frequency - centroid)**2 #How far away is the frequency from the centroid?
            numerator += magnitude * diff_sq #And how strong is it?
            denominator += magnitude

        return (numerator/denominator)**0.5 if not np.isnan((numerator/denominator)**0.5) else 0

    spectral_spreads = list(map(lambda x: spread_for_frame(x[0], x[1]), zip(spectrogram, centroids)))
    return spectral_spreads

def get_spectral_skewness(spectrogram, centroids, spreads, n_fft, sampling_rate):

    def skewness_for_frame(frame, centroid, spread):
        numerator = 0
        denominator = 0
        for frequency in range(len(frame)):
            magnitude = frame[frequency]
            frequency = map_sinus_frequency_to_hz(frequency, n_fft, sampling_rate)
            diff_sq = (frequency - centroid)**3
            numerator += magnitude * diff_sq
            denominator += magnitude
        return numerator/(denominator * spread**3) if not np.isnan(numerator/(denominator * spread**3)) else 0

    spectral_skews = list(map(lambda x: skewness_for_frame(x[0], x[1], x[2]), zip(spectrogram, centroids, spreads)))
    return spectral_skews

def get_average_frequencies(spectrogram, n_fft, sampling_rate):

    averaged_frequencies = np.mean(spectrogram, axis=0)
    averaged_frequencies = np.log10(np.power(np.power(averaged_frequencies, 2), 0.5))
    freq_mag_pairs = []
    for i in range(len(averaged_frequencies)):
        freq_mag_pairs.append((i, averaged_frequencies[i]))

    freq_mag_pairs.sort(key=lambda x: x[1], reverse=True)
    magnitudes = [pair[1] for pair in freq_mag_pairs]
    plt.plot(magnitudes)
    plt.show()
    freq_mag_pairs = freq_mag_pairs[-(len(freq_mag_pairs)//2):]

    return [pair[0] for pair in freq_mag_pairs]

class Filter_type(Enum):
    band_stop = 0
    band_pass = 1
def band_filter(spectrogram, center_frequency, width, filter_type:Filter_type, raw = True, curve_exponent = 1):

    try:
        filtered_spectrogram = spectrogram.copy()

        frequency_band = (len(spectrogram[0])//2) if raw else len(spectrogram[0])
        window = wd.gaussian(width * 2, width)
        upper_reach = center_frequency + width
        lower_reach = center_frequency - width
        above_band = upper_reach > frequency_band
        below_band = lower_reach < 0

        if above_band:
            #print("More than frequency band")
            window = window[-(upper_reach - frequency_band)]

        if  below_band:
            #print("less than frequency band")
            #print("Lower reach: ", abs(lower_reach))
            window = window[abs(lower_reach):]

        if filter_type == Filter_type.band_stop:
            window = np.subtract(max(window), window)

        if curve_exponent != 1.0:
            window = np.power(window, curve_exponent)

        if not above_band:
            until_max = frequency_band - upper_reach
            window = np.append(window, [min(window) if filter_type == Filter_type.band_pass else max(window) for _ in range(until_max)], axis=0)

        if not below_band:
            until_bottom = lower_reach
            window = np.append([min(window) if filter_type == Filter_type.band_pass else max(window) for _ in range(until_bottom)], window, axis=0)

        if raw:
            window = np.append(window, window[::-1], axis=0)


        for frame_index in range(len(spectrogram)):
            frame = list(spectrogram[frame_index])
            before_magnitude = np.linalg.norm(frame)
            frame = list(np.multiply(frame, window))
            post_magnitude = np.linalg.norm(frame)

            magnitude_change = post_magnitude / before_magnitude

            frame = np.divide(frame, magnitude_change)

            filtered_spectrogram[frame_index] = frame

        return filtered_spectrogram
    except TypeError:
        return spectrogram


def root_mean_square_error(first, second):
    summed = 0

    for j in range(len(first)):
        if np.isnan(first[j]) or np.isnan(second[j]): continue
        summed = np.add(np.power(np.subtract(first[j], second[j]), 2), summed)
    return (summed/len(first))**0.5
def raw_spectrogram_to_samples(spectrogram, hop_size, window_size, n_samples_of_whole_audio):
    samples = np.zeros(n_samples_of_whole_audio)

    start_sample = 0
    for frame in spectrogram:
        new_samples = np.fft.ifft(frame)
        for s in range(window_size):
            if s + start_sample >= n_samples_of_whole_audio: continue
            samples[s + start_sample] += new_samples[s]
        start_sample += hop_size

    return samples

def get_average_spectrals(spectrogram, n_fft, sampling_rate):

    centroids = get_spectral_centroids(spectrogram, n_fft, sampling_rate)
    spreads = get_spectral_spread(spectrogram, centroids, n_fft, sampling_rate)
    skews = get_spectral_skewness(spectrogram, centroids, spreads, n_fft, sampling_rate)

    average_centroid = np.mean(centroids)
    average_spread = np.mean(spreads)
    average_skew = np.mean(skews)
    return average_centroid, average_spread, average_skew

def cost_function(current, target):
    ratios = np.divide(current, target)
    difference = np.abs(np.subtract(ratios, 1))
    magnitude = np.linalg.norm(difference)
    return magnitude

traffic_samples, sr = librosa.load("traffic-sound.mp3")

samples, sr = librosa.load("Homer reading.mp3")
traffic_samples = traffic_samples[:len(samples)]
cm = np.add(traffic_samples, samples*7)

sf.write("Combined.mp3", cm, sr)

plt.plot(samples)
plt.show()

window_size = 2048
overlap = 0.5
window = wd.hann(window_size)

windows = generate_windows(traffic_samples, window_size=window_size, overlap=overlap, shape=window)
cm_spc = make_spectrogram(windows, False)
img = specshow(cm_spc.T, y_axis='linear', x_axis='time', sr=sr, win_length=window_size, hop_length=round(window_size - window_size * overlap))
plt.title("Traffic")
plt.show()

windows = generate_windows(samples, window_size=window_size, overlap=overlap, shape=window)
cm_spc = make_spectrogram(windows, False)
img = specshow(cm_spc.T, y_axis='linear', x_axis='time', sr=sr, win_length=window_size, hop_length=round(window_size - window_size * overlap))
plt.title("Homer reading")
plt.show()

windows = generate_windows(cm, window_size=window_size, overlap=overlap, shape=window)
cm_spc = make_spectrogram(windows, False)
img = specshow(cm_spc.T, y_axis='linear', x_axis='time', sr=sr, win_length=window_size, hop_length=round(window_size - window_size * overlap))
plt.title("Combined")
plt.show()

windows = generate_windows(samples, window_size=window_size, overlap=overlap, shape=window)
gram = make_spectrogram(windows)
frequencies_to_be_removed = get_average_frequencies(gram, n_fft=2048, sampling_rate=sr)

targets = get_average_spectrals(gram, 2048/2, sr)


combined_samples, _ = librosa.load("Combined.mp3")
combined_windows = generate_windows(combined_samples, window_size=window_size, overlap=overlap, shape=window)
img = specshow(make_spectrogram(combined_windows, False).T, y_axis='linear', x_axis='time', sr=sr, win_length=window_size, hop_length=round(window_size - window_size * overlap))
plt.show()
print("Original root mean square", root_mean_square_error(samples, combined_samples))
steps = []
impacts = [cost_function(get_average_spectrals(make_spectrogram(combined_windows, False), 2048/2, sr), targets)]
best_gram = make_spectrogram(combined_windows, False)

def test_frequency(center_frequency, best_gram, sr, targets):
    test_gram = best_gram.copy()
    step = lambda x, y: band_filter(x, center_frequency, 10, Filter_type.band_stop, raw=y)
    test_gram = step.__call__(test_gram, False)
    averages = get_average_spectrals(test_gram, 2048 / 2, sr)
    impact = cost_function(averages, targets)  # if len(impacts) == 0 else cost_function(averages, targets) - impacts[-1]
    return step, impact



while True:

    step_scores = list(map(lambda y: test_frequency(y, best_gram, sr, targets), range(2, 2048//2, 5)))
    step_scores.sort(key = lambda x: x[1], reverse = False)
    best_step = step_scores[0][0]
    best_gram = best_step.__call__(best_gram, False)
    steps.append(best_step)
    impacts.append(step_scores[0][1])
    plt.plot(impacts)
    plt.show()

    print("Impact: ", impacts[-1])

    if impacts[-1] < 0.01 or len(impacts) >= 3 and np.abs(float(np.mean(impacts[-3:], axis=0) - impacts[-1])) < 0.01:
        break
img = specshow(best_gram.T, y_axis='linear', x_axis='time', sr=sr, win_length=window_size, hop_length=round(window_size - window_size * overlap))
plt.show()

resulting_averages = get_average_spectrals(best_gram, 2048 / 2, sr)
print("Resulting", resulting_averages)
print("Target", targets)
transformed_gram = make_spectrogram(combined_windows, keep_raw=True)
for step in steps:
    transformed_gram = step.__call__(transformed_gram, True)
transformed_samples = raw_spectrogram_to_samples(transformed_gram, round(window_size - window_size * overlap), window_size, len(samples))
sf.write("Output.mp3", transformed_samples, sr)
print("RMSE", root_mean_square_error(samples, transformed_samples))