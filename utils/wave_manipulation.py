from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
import numpy as np

def wave_read():
    return wavread(speech_file)

def wave_write(dir_name,freq, signal_data):
    wavwrite(dir_name, freq, signal_data)
    
def normalize(signal_sound_data):
    signal_sound_data = np.asarray(signal_sound_data,dtype=float)
    signal_sound_data_norm = signal_sound_data / 32768.0    
    return signal_sound_data_norm

def denormalize(signal_sound_data):
    signal_sound_data = signal_sound_data * 32768.0
    Y = np.asarray(signal_sound_data,dtype=np.int16)
    return Y

    
if __name__ == '__main__':
    pass