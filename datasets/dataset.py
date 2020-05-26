from scipy.io import wavfile
import numpy as np

def get_tedx(n=None):
    from scipy import signal

    fs, data = wavfile.read('recordings/tedx.wav')

    data = data[:n] if n is not None else data

    #new_samples = int(len(data)*8000/fs)

    ##data = (data[:, 0]+data[:, 1])/2
    #data = signal.resample(data, new_samples)

    #data = np.copy(data) - np.min(data)
    #data = data / np.max(data)
    data = data / np.max(np.abs(data))

    return data


def get_speech_like(window):
    samples = 1000
    t = np.linspace(0, 1, samples)

    signal1 = np.zeros_like(t)
    signal2 = np.zeros_like(t)

    frequencies = [110, 120, 130, 115, 125, 135]

    for frequency in frequencies[:3]:
        signal1 += np.sin(2 * np.pi * frequency * t)

    for frequency in frequencies[3:]:
        signal2 += np.sin(2 * np.pi * frequency * 2 * t)

    signal2 = np.concatenate((signal2[1:], [0]))

    min_frequency = np.min(frequencies)
    #window = np.ceil(samples / min_frequency).astype('int32')

    signal = np.concatenate((signal1, signal2))

    blocks = int(len(signal) / window)
    data = signal[:blocks * window] if blocks * window < len(signal) else np.concatenate(
        (signal, np.zeros(blocks * window - len(signal))))

    data = np.concatenate([data, data], axis=0)


    #y_train = np.asarray([data[i + window:i + window + 1] for i in range(len(data) - window - 1)])
    #y_train = np.asarray([data[i+1:i + window+1] for i in range(len(data) - window - 1)])
    #x_train = np.asarray([data[i:i + window] for i in range(len(data) - window - 1)])

    return data #x_train, y_train

