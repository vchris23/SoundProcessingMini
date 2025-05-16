import math
import numpy as np
from matplotlib import pyplot as plt

def plot_amplitude_response(filter):
    t = np.arange(2000) / 2000
    radians = 2 * np.pi * t
    frequencies = range(400)
    sinus_waves = []
    for i in frequencies:
        sinus_waves.append(list(np.sin(radians * i)))

    transformed_waves = list(map(filter, sinus_waves))
    amplitudes = np.max(np.abs(transformed_waves), axis=1)
    plt.plot([f * 10 for f in frequencies], amplitudes)
    plt.show()
def low_pass_filter(samples:list, delay:int, coefficients:list):
    filtered_samples = []
    for i in range(len(samples)):
        if i < delay:
            filtered_samples.append(samples[i])
            print(i, delay, "I < delay")
            continue
        current_coefficients = coefficients[:math.floor(i/delay)]
        summed_samples = []
        for z in range(len(current_coefficients)):
            summed_samples.append(samples[i - delay * z] * current_coefficients[z])
        filtered_samples.append(sum(summed_samples)/len(coefficients))

    return filtered_samples

def high_pass_filter(samples:list, delay:int, coefficients:list):
    filtered_samples = []
    low_pas = low_pass_filter(samples, delay, coefficients)
    for i in range(len(low_pas)):
        filtered_samples.append(samples[i] - low_pas[i])
    return filtered_samples

