import keras 
from keras.layers import *
from keras import models , losses , optimizers , metrics
import numpy as np
import win_unicode_console
win_unicode_console.enable()

import checkbox_dataloader as loader
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# For Single Checkbox
def CreatModel_Agree( inputsize ): 
    res = (inputsize)
    input = Input( res)
    x = Conv2D( 64 , (5,5) ,padding = 'same', activation = "relu")(input)
    x = Conv2D( 32 , (5,5) ,padding = 'same' , activation = "relu")(x)
    x = MaxPool2D( padding = "same" )(x)
    x = Flatten()(x)
    x = Dense( 128 , activation = 'relu' )(x)
    x = Dense( 64 , activation = 'relu' )(x)
    output = Dense( 1 , activation = 'sigmoid')(x)
    
    model = models.Model( input , output ) 
    model.compile( optimizer = optimizers.SGD(0.002) , loss = losses.binary_crossentropy , metrics = ['acc'])
    return model


## 여기서 생성된 데이터를 훈련시켜서 얻은 웨이트 파일을 가지고 스크립트에서 실행을 시키자. 모델의 인풋 모양 확인해보자. 
def Train_Agree():
    agreepath = r'F:\00_gitbio\OCR_bio\OCR_bio\data\agree_Labeled'
    notagreepath = r'F:\00_gitbio\OCR_bio\OCR_bio\data\agree_Labeled\gennotAgree'
    xs0 = loader.SimpleLoader(notagreepath)
    xs1 = loader.SimpleLoader(agreepath)

    xs0 = np.expand_dims( xs0 , axis = -1 )
    xs1 = np.expand_dims( xs1 , axis = -1 )
    xs = np.concatenate( (xs0 , xs1) , axis = 0 )
   
    ys0 = np.zeros( len(xs0) )
    ys1 = np.ones( len(xs1) )
    ys = np.concatenate( (ys0,ys1) , axis = 0)

    inputsize = xs0[0].shape
    xtrain , xtest , ytrain , ytest = train_test_split(xs , ys  , test_size = 0.2 )

    model = CreatModel_Checkbox(inputsize)
    outpath = r"check3\weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    ck = keras.callbacks.ModelCheckpoint(outpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    hist = model.fit(xtrain,ytrain , batch_size = 16 , epochs  = 100 , validation_split = 0.2 , callbacks = [ck,tb])

def validationProcess():
    agreepath = r'F:\00_gitbio\OCR_bio\OCR_bio\data\agree_Labeled'
    notagreepath = r'F:\00_gitbio\OCR_bio\OCR_bio\data\agree_Labeled\gennotAgree'
    xs0 = loader.SimpleLoader(notagreepath)
    xs1 = loader.SimpleLoader(agreepath)
    xs0 = np.expand_dims( xs0 , axis = -1 )
    xs1 = np.expand_dims( xs1 , axis = -1 )


    model = models.load_model("")

    res = model.predict(xs0)
    print(res)
    print("_"*8)
    res = model.predict(xs1)
    print(res)

try :
    Train_Agree()
    #validationProcess()
except:
    print()

print()

