import numpy as np
import math
import h5py
import pdb
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def global_normalization(data,global_mean, global_var):
    m_global_mean = np.mean(global_mean)
    m_global_var = np.mean(global_var)
    gmbias = np.zeros(len(data))
    data_new = np.zeros(data.shape)
     
    for batch in range(data.shape[0]):
        for i in range(data.shape[1]):
            data_new[batch,i] = (data[batch,i] - global_mean[i%350] ) / global_var[i%350]
    
    '''
    for batch in range(len(data)):
        
        mdata = data[batch, :257]
        m_mdata = np.mean(mdata)
        if m_mdata <= m_global_mean+m_global_var/2 and m_mdata >= m_global_mean-m_global_var/2:
            gmbias[batch]=0;
        elif m_mdata <= m_global_mean+m_global_var and m_mdata > m_global_mean+m_global_var/2:
            gmbias[batch]=m_mdata-(m_global_mean+m_global_var/2) 
        elif m_mdata > m_global_mean+m_global_var :
            gmbias[batch]  =  m_mdata - m_global_mean + m_global_var
        elif m_mdata < m_global_mean-m_global_var/2 and m_mdata >= m_global_mean-m_global_var:
            gmbias[batch]=m_mdata-(m_global_mean-m_global_var/2)  
        else:
           gmbias[batch]  = m_mdata-(m_global_mean-m_global_var)

        
        for i in range(257+93):
            if i > 257:
                data_new[batch,i] = (data[batch,i] - global_mean[i] ) / global_var[i]
            else:
                data_new[batch,i] = (data[batch,i] - global_mean[i] - gmbias[batch] ) / global_var[i] 
    ''' 
    return data_new, gmbias
    
def inverse_norm(data,stat_file):
    '''
    hf = h5py.File(stat_file,'r') 
    global_mean = hf.get('global_mean')[()]
    global_var = hf.get('global_var')[()]
    
    for batch in range(data.shape[0]):
        for i in range(data.shape[1]):
            data[batch,i] = data[batch,i]  global_var[i%350] + global_mean[i%350]
    '''
    dataout_p1 = data[:,:257]
    dataout_p2 = data[:, -257:]
    return dataout_p1, dataout_p2 , data

def IBMpp(dataout_p1, dataout_p2):
    
    irm = np.sqrt(np.exp(dataout_p1))
    '''
    irm =np.zeros(dataout_p1.shape)
    irm[0,:] = dataout_p1[0,:]
    for t in range(1,dataout_p1.shape[0]):
        irm[t,:] = (dataout_p1[t,:] + dataout_p2[t-1,:]) / 2.
    irm = np.sqrt(np.exp(irm))
    '''
    return irm

def correct_gain(irm, noisy, stat_file):
    
    return irm
    '''
    irm_new = irm
    noisy_t = np.transpose(noisy)
    #for i in range(irm.shape[0]):
    for j in range(irm.shape[1]):
        if np.mean(irm[1:20,j])< 0.2 :
            irm_new[:,j] = irm[:,j] *0.001
        elif np.mean(irm[1:20,j])< 0.4 :
            irm_new[:,j] = irm[:,j] * .05
        else:
            irm_new[:,j]  = irm[:,j]
    return irm_new    
    '''
    '''
    hf = h5py.File(stat_file,'r') 
    global_mean = hf.get('global_mean')[()]
    global_var = hf.get('global_var')[()]
    
    noisy_t = np.transpose(noisy)
    mean_noisy = np.mean(noisy_t,axis = 1)
    var_noisy = np.var(noisy_t,axis = 1)
    
    mean_irm = np.mean(irm[:,:12], axis = 1)
    var_irm = np.var(irm[:,:12],axis = 1)
    DNNenh_f = irm
    #pdb.set_trace()
    for d in range(irm.shape[1]):
        DNNenh_f[:,d] = ((irm[:,d] -mean_irm ) /var_irm ) var_noisy + mean_noisy

    return DNNenh_f
    '''

    
if __name__ == '__main__':
    pass