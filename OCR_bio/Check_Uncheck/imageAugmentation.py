from keras.preprocessing.image import *
from checkbox_dataloader import *
import numpy as np

##
size = (32,32)
checkpath = r'F:\00_gitbio\OCR_bio\OCR_bio\data\checksplited\col5\1'
uncheckpath = r'F:\00_gitbio\OCR_bio\OCR_bio\data\checksplited\col5\0'
xs0 = SimpleLoader_FixedSize(uncheckpath,size)
#xs1 = SimpleLoader_FixedSize(checkpath,size)
xs0 = np.expand_dims( xs0 , axis = -1 )
#xs1 = np.expand_dims( xs1 , axis = -1 )
#xs = np.concatenate( (xs0 , xs1) , axis = 0 )

#ys0 = np.zeros( len(xs0) )
#ys1 = np.ones( len(xs1) )
#ys = np.concatenate( (ys0,ys1) , axis = 0)

##

datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

i = 0 
for batch in datagen.flow( xs0 , save_to_dir='data\\checksplited\\uncheckGen' ,save_prefix='col5', save_format='png' ):
    i += 1
    if i > 50:
        break






