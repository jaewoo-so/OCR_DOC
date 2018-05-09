import keras 
from keras.layers import *
from keras import models , losses , optimizers , metrics
import numpy as np
import win_unicode_console
win_unicode_console.enable()

import checkbox_dataloader as loader
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# For 5column 
def CreatModel( inputsize ): 
    res = (inputsize + (1,))
    input = Input( res)
    x = Conv2D( 32 , (5,5) ,padding = 'same', activation = "relu")(input)
    x = Dropout(0.3)(x)
    x = Conv2D( 32 , (5,5) ,padding = 'same' , activation = "relu")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D( padding = "same" )(x)
    x = Flatten()(x)
    x = Dense( 256 , activation = 'relu' )(x)
    x = Dense( 128 , activation = 'relu' )(x)
    output = Dense( 5 , activation = 'softmax')(x)
    
    model = models.Model( input , output ) 
    model.compile( optimizer = optimizers.Adam() , loss = losses.categorical_crossentropy , metrics = ['acc'])
    return model

# For Single Checkbox
def CreatModel_Checkbox( inputsize ): 
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
def checkAnduncheck():
    checkpath = r'F:\00_gitbio\OCR_bio\OCR_bio\data\checksplited\checkGen'
    uncheckpath = r'F:\00_gitbio\OCR_bio\OCR_bio\data\checksplited\uncheckGen'
    xs0 = loader.SimpleLoader(uncheckpath)
    xs1 = loader.SimpleLoader(checkpath)
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
    tb = keras.callbacks.TensorBoard(r"logs")

    hist = model.fit(xtrain,ytrain , batch_size = 16 , epochs  = 1000 , validation_split = 0.2 , callbacks = [ck,tb])

def validationProcess():
    checkpath = r'F:\00_gitbio\OCR_bio\OCR_bio\data\checksplited\1'
    uncheckpath = r'F:\00_gitbio\OCR_bio\OCR_bio\data\checksplited\0'
    xs0 = loader.SimpleLoader(uncheckpath)
    xs1 = loader.SimpleLoader(checkpath)
    xs0 = np.expand_dims( xs0 , axis = -1 )
    xs1 = np.expand_dims( xs1 , axis = -1 )


    model = models.load_model("check2\\weights.33-0.43.hdf5")

    res = model.predict(xs0)
    print(res)
    print("_"*8)
    res = model.predict(xs1)
    print(res)





def col5():
    # Part 2 - Fitting the CNN to the images
    col5path = r'F:\00_gitbio\OCR_bio\OCR_bio\data\survey_result_Labeled\col5'
    xs , labels , size = loader.LoadCol5(col5path)
    
    xtrain , xtest , ytrain , ytest = train_test_split(xs , labels  , test_size = 0.2 )

    model = CreatModel(size)
    #callback
    es =keras.callbacks.EarlyStopping( monitor = 'val_loss' , min_delta = 1.0 , patience = 2 )
    outpath = r"check\weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    ck = keras.callbacks.ModelCheckpoint(outpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    tb = keras.callbacks.TensorBoard(r"logs") 

    datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range = 0.1 )



    
    hist = model.fit_generator( datagen.flow(xtrain, ytrain, batch_size=8),
                               steps_per_epoch = 100 ,
                               epochs = 300,
                               validation_data = (xtest,ytest),
                               validation_steps = 1, callbacks = [ ck , tb])

    #tep = int(len(ytrain) / 4)
    #hist = model.fit(xtrain, ytrain, batch_size=8,epochs = 3, callbacks = [ ck , tb])
    

    score , res = model.evaluate(xtest , ytest )

    print('Test score:', score)
    print('Test accuracy:', acc)

try :
    checkAnduncheck()
    #validationProcess()
except:
    print()

print()