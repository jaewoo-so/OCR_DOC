#from Data_Loader import HangulLoader as ld

import os
import keras
from keras.layers import *
from keras import models , applications , callbacks , optimizers , losses
import numpy as np
from sklearn.model_selection import train_test_split
from functools import *

from Data_Loader import HangulLoader as hanLoader
from C1_model import CreateC1

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
xs , ys , n_cls = hanLoader().GetImages_Labels( dirpath, labelspath , isGray = True )

# 데이터 클래스 사이즈 줄이기 
# 모델 아키텍쳐 조금 바꿔보기


# load model 
print("x shape : {}".format( xs.shape ))
print("y shape : {}".format( ys.shape ))

input = Input((64,64,1))

n_out1 = 32
x = Conv2D(n_out1,(3,3) ,padding = 'same',activation = 'relu')(input)
#x = Dropout(0.5)(x)
#x = Conv2D(n_out1,(3,3),padding = 'same', activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = MaxPool2D((2,2))(x)

n_out2 = 64
x = Conv2D(n_out2,(3,3) ,padding = 'same',activation = 'relu')(x)
#x = Dropout(0.5)(x)
#x = Conv2D(n_out2,(3,3),padding = 'same', activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = MaxPool2D((2,2))(x)

n_out3 = 128
x = Conv2D(n_out3,(3,3) ,padding = 'same',activation = 'relu')(x)
#x = Dropout(0.5)(x)
#x = Conv2D(n_out3,(3,3),padding = 'same', activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = MaxPool2D((2,2))(x)

n_out4 = 256
x = Conv2D(n_out4,(3,3) ,padding = 'same',activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Conv2D(n_out4,(3,3),padding = 'same', activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = MaxPool2D((2,2))(x)


x = Flatten()(x)
x = Dense( 1024 , activation = "relu" )(x)
x = Dense( 1024 , activation = "relu" )(x)
lastLayer = Dense( n_cls , activation = 'softmax' )(x)

#classifier = [ dfeature , flat , dense1 , dense2 , lastLayer  ]
#output = reduce( lambda f,s : s(f) , classifier)
model_cnn = models.Model( input , lastLayer )

#callback 
outpath = r"check\weights.{epoch:02d}-{val_loss:.2f}.hdf5"
ck = keras.callbacks.ModelCheckpoint(outpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
tb = keras.callbacks.TensorBoard(r"logs")

model_cnn.compile( loss = losses.categorical_crossentropy , optimizer = optimizers.Adam(0.002) , metrics = ['acc'] )

model_cnn.summary()

#K.clear_session()
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=5,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(xs)

hist = model_cnn.fit_generator(datagen.flow(xs, ys, batch_size=32),
                    steps_per_epoch=len(xs) / 32, epochs=10000)

print()













