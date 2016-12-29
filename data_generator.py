#!/usr/bin/python

from __future__ import division
import numpy as np
import math
from scipy.special import *
# from audiolab import Sndfile, Format
import argparse
import sys
import pdb 

np.seterr('raise')


def generate(x, Srate, noise_frames=6, Slen=0):
    if Slen == 0:
        Slen = int(math.floor(0.02 * Srate))

    if Slen % 2 == 1:
        Slen = Slen + 1

    PERC = 50
    len1 = math.floor(Slen * PERC / 100)
    len2 = int(Slen - len1)

    win = np.hanning(Slen)
    win = win * len2 / np.sum(win)
    nFFT = 2 * Slen

    x_old = np.zeros(len1)
    Xk_prev = np.zeros(len1)
    Nframes = int(math.floor(len(x) / len2) - math.floor(Slen / len2))
    xfinal = np.zeros(Nframes * len2)

    noise_mean = np.zeros(nFFT)
    for j in range(0, Slen*noise_frames, Slen):
        noise_mean = noise_mean + np.absolute(np.fft.fft(win * x[j:j + Slen], nFFT, axis=0))
    noise_mu2 = noise_mean / noise_frames ** 2

    aa = 0.98
    mu = 0.98
    eta = 0.15
    ksi_min = 10 ** (-25 / 10)
    X_train_sig = np.array([])
    X_train_spec = np.array([])
    
    for k in range(0, Nframes*len2, len2):
        insign = win * x[k:k + Slen]

        spec = np.fft.fft(insign, nFFT, axis=0)
        ## log is better
        ## energy normalizaation

        sig = np.absolute(spec)
        sig2 = sig ** 2

        gammak = np.minimum(sig2 / noise_mu2, 40)

        if Xk_prev.all() == 0:
            ksi = aa + (1 - aa) * np.maximum(gammak - 1, 0)
        else:
            ksi = aa * Xk_prev / noise_mu2 + (1 - aa) * np.maximum(gammak - 1, 0)
            ksi = np.maximum(ksi_min, ksi)

        log_sigma_k = gammak * ksi/(1 + ksi) - np.log(1 + ksi)
        vad_decision = np.sum(log_sigma_k)/Slen
        if (vad_decision < eta):
            noise_mu2 = mu * noise_mu2 + (1 - mu) * sig2

        A = ksi / (1 + ksi)
        vk = A * gammak
        ei_vk = 0.5 * expn(1, vk)
        hw = A * np.exp(ei_vk)

        sig = sig * hw
        Xk_prev = sig ** 2
        
        if X_train_sig.shape[0] == 0:
            X_train_sig = sig 
            X_train_spec = spec
        else:
            X_train_sig = np.vstack((X_train_sig,sig))
            X_train_spec = np.vstack((X_train_spec,spec))
            
    return X_train_sig ,X_train_spec

if __name__ == '__main__':
    pass      