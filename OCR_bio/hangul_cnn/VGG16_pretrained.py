#from Data_Loader import HangulLoader as ld

import os
import keras
from keras.layers import *
from keras import models , applications , callbacks , optimizers , losses
import numpy as np
from sklearn.model_selection import train_test_split
from functools import *

from Data_Loader import HangulLoader as hanLoader

import win_unicode_console
win_unicode_console.enable()

import keras.backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))



# data x ,y 
labelspath = r"data\hangul-images-data\labels-map.csv"
dirpath = r"data\hangul-images-data\img"
xs , ys , n_cls = hanLoader().GetImages_Labels( dirpath, labelspath )
# load model 
print("x shape : {}".format( xs.shape ))
print("y shape : {}".format( ys.shape ))

deepFeature = applications.VGG16(include_top = False , input_shape = (64,64,3) )
deepFeature.trainable = False
input = deepFeature.input

h = deepFeature.output

flat = Flatten()
dense1 = Dense( 4096 , activation = "relu" )
dense2 = Dense( 3096 , activation = "relu" )
lastLayer = Dense( n_cls , activation = "sigmoid"  )

classifier = [ h , flat , dense1 , dense2 , lastLayer  ]
output = reduce( lambda f,s : s(f) , classifier)
model_cnn = models.Model( input , output )

#callback 
outpath = r"check\weights.{epoch:02d}-{val_loss:.2f}.hdf5"
ck = keras.callbacks.ModelCheckpoint(outpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
tb = keras.callbacks.TensorBoard(r"logs")

model_cnn.compile( loss = losses.categorical_crossentropy , optimizer = optimizers.Adam() , metrics = ['acc'] )


hist = model_cnn.fit( xs , ys , batch_size = 32 , epochs  = 10 , validation_split = 0.2 , callbacks = [ck,tb])

print()















