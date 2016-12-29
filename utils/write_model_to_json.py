from keras.models import load_model
import simplejson
import pdb 
import cPickle as pickle
import os
import h5py
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop,Adam

path = '../model/model_json.json'
def main():        
    layer1_dimention = 4500
    layer2_dimention = 3500
    layer3_dimention = 3000
    layer4_dimention = 2000
    numCep = 40
    frame_step = 80
    frame_len = 1024
    fft_len = 1024
    no_frames = 6
    input_dim = (no_frames+1) * (numCep + fft_len) # +1 is for noise aware system
    output_dim = fft_len

    model = Sequential()
    model.add(Dense(input_dim= input_dim, output_dim = layer1_dimention, init='glorot_uniform' ) )
    model.add(Activation('sigmoid'))
    model.add(Dropout(.2))
    model.add(Dense(input_dim=layer1_dimention, output_dim = layer2_dimention, init='glorot_uniform' ) )
    model.add(Activation('sigmoid'))
    model.add(Dropout(.2))
    model.add(Dense(input_dim=layer2_dimention, output_dim = layer3_dimention, init='glorot_uniform' ) )
    model.add(Activation('sigmoid'))
    model.add(Dropout(.2))
    model.add(Dense(input_dim=layer3_dimention, output_dim = layer4_dimention, init='glorot_uniform' ) )
    model.add(Activation('sigmoid'))
    model.add(Dropout(.2))
    model.add(Dense(input_dim=layer4_dimention, output_dim = output_dim, init='glorot_uniform' ))
    model.add(Activation('linear'))

    RMS = RMSprop(lr=0.001, rho=0.9, epsilon=1e-12)
    model.compile(loss = 'mse', optimizer = RMS)
    model.summary()            

    model_json = model.to_json()
    with open(path, 'w') as json_file:
        json_file.write(simplejson.dumps(simplejson.loads(model_json), indent = 4, sort_keys=True))
        
    #weight = model.get_weights()
    #pickle.dump(weight, open(hdf_file[:-3] + '.pkl', 'wb' ) )
 
if __name__ == '__main__':
    main()  