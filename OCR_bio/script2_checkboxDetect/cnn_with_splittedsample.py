
import keras 
from keras.layers import *
from keras import models , losses , optimizers , metrics
import numpy as np
import win_unicode_console
win_unicode_console.enable()

import checkbox_dataloader as loader
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def CreatModel( inputsize ): 
    res = (inputsize + (1,))
    input = Input( res)
    x = Conv2D( 32 , (3,3) ,padding = 'same', activation = "relu")(input)
    x = MaxPool2D( padding = "same" )(x)
    x = Conv2D( 64 , (3,3) ,padding = 'same' , activation = "relu")(x)
    x = MaxPool2D( padding = "same" )(x)
    x = Flatten()(x)
    x = Dense( 256 , activation = 'relu' )(x)
    x = Dropout(0.3)(x)
    x = Dense( 128 , activation = 'relu' )(x)
    output = Dense( 5 , activation = 'softmax')(x)
    
    model = models.Model( input , output ) 
    model.compile( optimizer = optimizers.Adam() , loss = losses.categorical_crossentropy , metrics = ['acc'])
    return model

def col5():
    # Part 2 - Fitting the CNN to the images
    col5path = r'F:\00_gitbio\OCR_bio\OCR_bio\data\survey_result_Labeled\col5'
    xs , labels , size = loader.LoadCol5(col5path)
    xsplited , ysplited = loader.DataSpliter(xs , labels , 5 , xs[0].shape )
    


    
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
    col5()
except:
    print()

print()