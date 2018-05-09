import keras
from keras.layers import *
from keras import models , losses , optimizers , metrics
import numpy as np
import win_unicode_console

win_unicode_console.enable()


input = Input((64,64,3))

x = Conv2D( 64 , (3,3) , activation = "relu")(input)
x = MaxPool2D( padding = "same" )(x)
x = Flatten()(x)
x = Dense( 128 , activation = 'sigmoid' )(x)
output = Dense( 3 , activation = 'softmax')(x)

model_ver1 = models.Model( input , output ) 
model_ver1.compile( optimizer = optimizers.Adam() , loss = losses.categorical_crossentropy , metrics = ['acc'])

#callback
es =keras.callbacks.EarlyStopping( monitor = 'val_loss' , min_delta = 1.0 , patience = 2 )
outpath = r"check\weights.{epoch:02d}-{val_loss:.2f}.hdf5"
ck = keras.callbacks.ModelCheckpoint(outpath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
tb = keras.callbacks.TensorBoard(r"logs") 


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2 )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data\\images_out',
                                                 target_size = (64, 64),
                                                 color_mode = "rgb",
                                                 batch_size = 32 )



test_set = test_datagen.flow_from_directory('data\\images_out_test',
                                            target_size = (64, 64),
                                            color_mode = "rgb",
                                            batch_size = 32)

hist = model_ver1.fit_generator(training_set,
                         steps_per_epoch = 10,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 100, callbacks = [ ck , tb])


print()

## Test Section
'''
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/Users/hyunmin/Downloads/project-handwriting/images_out_test/1/8.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = '개인정보 수집 및 동의서'
else:
    prediction = '유전자 검사 동의서'
    
print(prediction)
'''
