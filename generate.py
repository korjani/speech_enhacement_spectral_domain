import pdb
import glob
import os
import numpy as np
from keras.models import load_model
import h5py
from scipy.fftpack import ifft
from scipy.signal import hamming
from sklearn import preprocessing
from pydub import AudioSegment
import sys
sys.path.append(os.path.abspath('utils'))
import wave_manipulation as manipulate
import feature_extraction as mfcc
import librosa 
import parameters as parameter
import argparse
import generate_model


NB_EPOCH = parameter.NB_EPOCH 
BATCH_SIZE = parameter.BATCH_SIZE 
no_frames = parameter.no_frames 

numCep = parameter.numCep
frame_len = parameter.frame_len 
fft_len = parameter.fft_len 

input_dim = (no_frames+1) * (numCep + fft_len/2 + 1) # +1 is for noise aware system
output_dim = (fft_len/2 +1) +numCep
noise_aware_frame_lenght = parameter.noise_aware_frame_lenght 
SNR = parameter.SNR 

MODEL_FILE = parameter.RESULT_MODEL
SPEECH_FILE = parameter.TEST_SPEECH_FILE
NOISE_FILE =  parameter.TEST_NOISE_FILE
RESULT_DIR = parameter.RESULT_DIR

frame_step = 1

def get_arguments():
    parser = argparse.ArgumentParser(description='read h5py model and save the model in a json file and weights in a pickle file')
    parser.add_argument('--model_file', type=str, default=MODEL_FILE,
                        help='The directory containing the h5df model (model + weights).')
    parser.add_argument('--noisy_file', type=str, default=False,
                        help='noisy file.')
    parser.add_argument('--speech_file', type=str, default=SPEECH_FILE,
                        help='Speech file.')
    parser.add_argument('--noise_file', type=str, default=NOISE_FILE,
                        help='noise file.')
    parser.add_argument('--snr', type=int, default=SNR,
                        help='SNR of noisy data (how much noise is added to data.')
    parser.add_argument('--result_dir', type=str, default=RESULT_DIR,
                        help='directory to write result.')
    return parser.parse_args()

def validate_arg(arg):
    try:
        os.path.isfile(arg)
        print('file exists: %s' %arg)
    except:
        print('file not found: %s' %arg)
        return
    try:
        os.access(arg, os.R_OK)
        print('file is readable: %s' %arg)
    except:
        print('file is not readable: %s' %arg)
        return
    
    

def main():
    args = get_arguments()

    ## read model
    try:
        model = generate_model.generate()
    except:
        print('cant read the model!' )
        return
       
    ## load weights
    try:
        model.load_weights(args.model_file)
    except:
        print('no weight!' )
        return
        
    ## validate wave file 
    try:
        validate_arg(args.speech_file)
        validate_arg(args.noise_file)
    except ValueError as e:
        print("wave file is not available:")
        print(str(e))
        return  



    X_train = np.array([])
    Y_train = np.array([])  
     
    head, tail = os.path.split(args.speech_file)
    fs_signal, signal_sound_data = manipulate.wavread(args.speech_file)
    signal_sound = AudioSegment.from_file(args.speech_file)

    head_n, tail_n = os.path.split(args.noise_file)
    fs_noise, noise_sound_data = manipulate.wavread(args.noise_file)
    noise_sound = AudioSegment.from_file(args.noise_file)

    
    dB = signal_sound.dBFS - noise_sound.dBFS - SNR
    noise_sound += dB # adjust dB for noise relative to sound
    noise_sound_data = noise_sound.get_array_of_samples()

    rand_start = 0 #np.random.randint(len(noise_sound_data)- len(signal_sound_data))

    combined = signal_sound_data + noise_sound_data[rand_start: rand_start+ len(signal_sound_data)]
    noisy_data = combined.astype(np.int16)

    tmp_wav_path = 'result/tmp2' + '.wav'
    manipulate.wave_write(tmp_wav_path, fs_signal, noisy_data)
    
    ### ####  noisy wave file
    wav_n, sr = librosa.load(tmp_wav_path, sr=fs_signal, mono=True)
      
    ## log-power spectral
    stftMat = librosa.core.stft(wav_n, win_length=frame_len , n_fft=fft_len, hop_length=frame_step, center = True)
    ## mfcc
    D = np.abs(stftMat)**2
    log_power_spec = np.log(D)
    Mel = librosa.feature.melspectrogram(S=D)
    mfcc_noisy = librosa.feature.mfcc(wav_n, S=Mel, sr=sr, n_mfcc=numCep)
    log_power_spec_noisy = np.transpose(log_power_spec)
    ceps_noisy = np.transpose(mfcc_noisy)
    
    features_noisy = np.append(log_power_spec_noisy, ceps_noisy,axis=1)
    X_train = np.array([np.hstack(features_noisy[L + 1 - no_frames/2: L +2 +no_frames/2 , :]) for L in range(no_frames/2 -1, len(features_noisy)-no_frames/2-1) ])

    ## noise aware training     
    first_frame_noise = features_noisy[:noise_aware_frame_lenght, :]
    noise_aware = np.mean(first_frame_noise, axis=0) 
    aware = np.dot(np.reshape(np.ones(len(X_train)),(-1,1)) ,np.reshape(noise_aware, (1,-1)) )
    X_train = np.hstack((X_train, aware))

    print('loading model...')
    ## predict
    Y_pred = model.predict(X_train)
    
    ## ###########check if the coding and deconsing is the same!
    #Y_train = X_train[:, no_frames/2 *(numCep + fft_len/2 +1) : no_frames/2 *(numCep + fft_len/2 +1) +output_dim ]
    #Y_pred = Y_train
    ## ###########
    
    stat_file = ''
    dataout_p1, dataout_p2, dataout = mfcc.inverse_norm(Y_pred, stat_file)
    inv_dataout_p1 = np.transpose(dataout_p1)
    inv_dataout_p2 = np.transpose(dataout_p2)
    irm = mfcc.IBMpp(inv_dataout_p1, inv_dataout_p2)
    corrected_irm = mfcc.correct_gain(irm ,np.transpose(log_power_spec_noisy[no_frames/2 :-no_frames/2+1,: ]), stat_file)


    ## write results
    ## if Y_pred == Y_train then check if np.abs(stftMat[:,no_frames/2:-no_frames/2+1]) == irm
    phase = np.exp(1j * np.angle(stftMat[:,no_frames/2:-no_frames/2+1]))
    R = np.multiply(corrected_irm , phase ) 

    reverse = librosa.core.istft(R, win_length=frame_len , hop_length=frame_step, center=True )
    y_out = librosa.util.fix_length(reverse, len(signal_sound_data))

    librosa.output.write_wav(args.result_dir + 'result_' + tail[:-4] + '_' + tail_n[:-4] + '_' + str(SNR) +  'db.wav', y_out, fs_signal, norm=False)
   
    # write noisy
    noisy_data = np.asarray(noisy_data,dtype=np.int16)
    manipulate.wave_write(args.result_dir + 'noisy_' + tail[:-4] + '_' + tail_n[:-4] + '_' + str(SNR) + 'db.wav', fs_signal, noisy_data)     
    manipulate.wave_write(args.result_dir + 'signal_' + tail[:-4] + '.wav', fs_signal, signal_sound_data)                
     
if __name__ == '__main__':
    main() 
     