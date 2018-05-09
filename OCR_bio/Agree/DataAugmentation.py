import numpy as np
import cv2
import os
from functools import *
from keras.preprocessing import image
    
basepath = r"data\agree_Labeled"
notagreePath = os.path.join(basepath , "0")
agreePath = os.path.join(basepath , "1")

saveagreepath = os.path.join(basepath , "genAgree")
savenotagreepath = os.path.join(basepath , "gennotAgree")

if not os.path.exists(saveagreepath):
    os.mkdir(saveagreepath)

if not os.path.exists(savenotagreepath):
    os.mkdir(savenotagreepath)


imggen = image.ImageDataGenerator( 
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2 ,
    horizontal_flip = True , 
    fill_mode = 'nearest')

def extractMargin(img,margin):
    w , h , c = None , None , None
    if len(img.shape) == 3 :
        h , w,c= img.shape
    else: 
        h , w = img.shape

    xstart = margin
    xend = w-margin
    ystart = margin
    yend = h - margin
    img = img[ystart:yend , :]
    return img


def getFiles(path):
    filenameList = []
    for root , _ , files in os.walk(path):
        for filepath in files:
            filename = os.path.join(root, filepath)
            filenameList.append(filename)
    return filenameList

x1 = getFiles( agreePath) 
x0 = getFiles(  notagreePath) 

imglist1 = map( lambda x :  extractMargin(cv2.imread(x , cv2.IMREAD_GRAYSCALE),4) , x1 )
imglist0 = map( lambda x :  extractMargin(cv2.imread(x , cv2.IMREAD_GRAYSCALE),4) , x0 )

imglist1 = map(lambda x :  cv2.resize(x, (32,32)) , imglist1 )
imglist0 = map(lambda x :  cv2.resize(x , (32,32)) , imglist0)

imglist1 = list(map( lambda x : cv2.threshold( x , 170 , 255 , cv2.THRESH_BINARY_INV)[-1] , imglist1))
imglist0 = list(map( lambda x : cv2.threshold( x , 170 , 255 , cv2.THRESH_BINARY_INV)[-1] , imglist0))


imglist1 = np.asarray( [ np.expand_dims(img , axis = -1) for img in imglist1])
imglist0 = np.asarray( [ np.expand_dims(img , axis = -1) for img in imglist0])

i = 0 
for img in imggen.flow( imglist1 , batch_size = 10 , save_to_dir = saveagreepath , save_prefix = "checked" , save_format = "png"):
    i += 1
    if i == 200:
        break
    
i = 0
for img in imggen.flow( imglist0 , batch_size = 10, save_to_dir = savenotagreepath , save_prefix = "unchecked" , save_format = "png"):
    i += 1
    if i == 200 : 
        break







