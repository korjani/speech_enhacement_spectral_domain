import pdb
import glob
import os
import numpy as np
import h5py
from keras.models import load_model
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import argparse
from pydub import AudioSegment
import sys
import numpy as np
from scipy.signal import hamming
from sklearn import preprocessing
sys.path.append(os.path.abspath('utils'))
import wave_manipulation as manipulate
import generate_model
import feature_extraction as mfcc
import librosa
import parameters as parameter


MODEL_WEIGHT = parameter.MODEL_WEIGHT 
SPEECH_FILE = parameter.SPEECH_FILE 
NOISE_FILE = parameter.NOISE_FILE 
RESULT_MODEL = parameter.RESULT_MODEL 
NB_EPOCH = parameter.NB_EPOCH 
BATCH_SIZE = parameter.BATCH_SIZE 
no_frames = parameter.no_frames 

numCep = parameter.numCep
frame_step = parameter.frame_step 
frame_len = parameter.frame_len 
fft_len = parameter.fft_len 

input_dim = (no_frames+1) * (numCep + fft_len/2 +1) # +1 is for noise aware system
output_dim = 2* (fft_len/2 +1) +numCep

noise_aware_frame_lenght = parameter.noise_aware_frame_lenght 

MIN_SNR = parameter.MIN_SNR 
MAX_SNR = parameter.MAX_SNR 

SNR = parameter.SNR 

X_train = np.array([])
Y_train = np.array([])  



def get_arguments():
    parser = argparse.ArgumentParser(description='speech enhancement in spectral domian')

    parser.add_argument('--model_weight', type=str, default=MODEL_WEIGHT,
                        help='The directory containing the h5df weights.')
    parser.add_argument('--speech_file', type=str, default=SPEECH_FILE,
                        help='Speech files.')
    parser.add_argument('--noise_file', type=str, default=NOISE_FILE,
                        help='Noise files.')
    parser.add_argument('--nb_epoch', type=int, default=NB_EPOCH,
                        help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='batch size')
    parser.add_argument('--min_snr', type=int, default=MIN_SNR,
                        help='minimum of SNR signal to noise')
    parser.add_argument('--max_snr', type=int, default=MAX_SNR,
                        help='maximum of SNR signal to noise')                        
    parser.add_argument('--result_model', type=str, default=RESULT_MODEL,
                        help='directory to write the trained model.')
    return parser.parse_args() 
 
 
def data_genertor():
    args = get_arguments()  
    while True:
        for fullpath_noise in glob.iglob(args.noise_file): 
            fs_noise, noise_sound_data = manipulate.wavread(fullpath_noise)
            noise_sound = AudioSegment.from_file(fullpath_noise)
            
            for fullpath in glob.iglob(args.speech_file): 
                fs_signal, signal_sound_data = manipulate.wavread(fullpath)
                signal_sound = AudioSegment.from_file(fullpath)
            

                
                #SNR = np.random.randint(min_snr,max_snr)
                dB = signal_sound.dBFS - noise_sound.dBFS - SNR
                noise_sound += dB # adjust dB for noise relative to sound
                noise_sound_data = noise_sound.get_array_of_samples()
                
                rand_start = np.random.randint(len(noise_sound_data)- len(signal_sound_data))
                # check the lenght of signal and noise , assume len(noise) > len(signal)

                combined = signal_sound_data + noise_sound_data[rand_start: rand_start+ len(signal_sound_data)]
                noisy_data = combined.astype(np.int16)
                noisy_data = np.asarray(noisy_data,dtype=np.int16)
                
                tmp_wav_path = 'result/tmp' + '.wav'
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
                features_noisy_not_norm = np.append(log_power_spec_noisy, ceps_noisy,axis=1)
                
                ## ####signal
                wav_s, sr = librosa.load(fullpath, sr=fs_signal, mono=True)
                ## log-power spectral
                stftMat_signal = librosa.core.stft(wav_s, win_length=frame_len , n_fft=fft_len, hop_length=frame_step, center = True)
                ## mfcc
                D_signal = np.abs(stftMat_signal)**2
                log_power_spec_signal = np.log(D_signal)
                Mel_signal = librosa.feature.melspectrogram(S=D_signal)
                mfcc_signal = librosa.feature.mfcc(wav_s, S=Mel_signal, sr=sr, n_mfcc=numCep)
                log_power_spec_signal = np.transpose(log_power_spec_signal)
                ceps_signal = np.transpose(mfcc_signal)
                features_clean_not_norm = np.append(log_power_spec_signal, ceps_signal,axis=1)
                
                ## normalized
                features_noisy = features_noisy_not_norm 
                features_clean = features_clean_not_norm 
                
                X_train = np.array([np.hstack(features_noisy[L + 1 - no_frames/2: L +2 +no_frames/2 , :]) for L in range(no_frames/2 -1, len(features_noisy)-no_frames/2-1) ])
                Y_train = np.array([np.hstack(features_clean[L + 1 - no_frames/2: L +2 +no_frames/2 , :]) for L in range(no_frames/2 -1, len(features_clean)-no_frames/2-1) ])
                Y_train = Y_train[:, no_frames/2 *(numCep + fft_len/2 +1) : no_frames/2 *(numCep + fft_len/2+1) +output_dim ]
                
                ## noise aware training 
                first_frame_noise = features_noisy[:noise_aware_frame_lenght, :]
                noise_aware = np.mean(first_frame_noise, axis=0) 
                aware = np.dot(np.reshape(np.ones(len(X_train)),(-1,1)) ,np.reshape(noise_aware, (1,-1)) )
                X_train = np.hstack((X_train, aware))
                
                yield X_train , Y_train # yield

               
                
                

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
        model.load_weights(args.model_weight)
    except:
        print('no weight!' )
        
    ## validate wave file 
    try:
        validate_arg(args.speech_file)
    except ValueError as e:
        print("wave file is not available:")
        print(str(e))
        return        
        
    data = data_genertor()
    
    ## training
    checkpoint = ModelCheckpoint(args.result_model, monitor='val_acc', verbose=0, save_weights_only=False, save_best_only=False , mode='auto') 
    callbacks_list = [checkpoint]  
    nb_files =50000         
    print('Training model...')

    model.fit_generator(data,
                        nb_files,
                        nb_epoch=args.nb_epoch,
                        callbacks=callbacks_list,
                        verbose = 1)

                        
 
if __name__ == '__main__':
    main()  
                    
                    