import numpy as np
from keras import models
import cv2
import os
import csv

# data is single raw of each content



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
    img = img[ystart:yend , xstart : xend]
    return img

def Predict_Checkbox(imgGray, colNum , model):
    imgGray = cv2.resize( imgGray , (240, 38) )
    splitedimg = []
    h , w = imgGray.shape
    step = int(w / colNum)
    for k in range(colNum):
         start = k*step
         end = (k+1)*step
         splitedimg.append( extractMargin(imgGray[:,start : end ],1))

    
    result = []
    for imgsplited in splitedimg:
        _ , binary_x = cv2.threshold( imgsplited , 220  , 255 , cv2.THRESH_BINARY_INV )
        binary_x = extractMargin(binary_x,7)
        binary_x = cv2.resize(binary_x , (32,32) )
        binary_x = np.expand_dims(binary_x , axis = -1)
        binary_x = np.expand_dims(binary_x , axis =  0)
        res = model.predict(binary_x)
        result.append(res)

    pred = np.argmax(result)
    return pred

def Predict_Checkbox_Debug(imgGray, colNum , model , savepath):
    imgGray = cv2.resize( imgGray , (240, 38) )
    splitedimg = []
    h , w = imgGray.shape
    step = int(w / colNum)
    for k in range(colNum):
         start = k*step
         end = (k+1)*step
         splitedimg.append( extractMargin(imgGray[:,start : end ],1))

    
    result = []
    for i , imgsplited in enumerate( splitedimg ):
        _ , binary_x = cv2.threshold( imgsplited , 220  , 255 , cv2.THRESH_BINARY_INV )
        binary_x = extractMargin(binary_x,7)
        
        savename = os.path.join(savepath , str(i) + ".png")
        cv2.imwrite(savename,binary_x)
        
        binary_x = np.expand_dims(binary_x , axis = -1)
        binary_x = np.expand_dims(binary_x , axis =  0)

       
        res = model.predict(binary_x)
        result.append(res)

    pred = np.argmax(result)
    return pred


def main_col(colnum):
    modelpath = r'Check_Uncheck\CheckUncheckModel.hdf5'
    datapath = r'data\survey_result_Labeled\col' + str(colnum)
    
    for root , subs , files in os.walk(datapath):
        if len(subs) !=  0:
            continue

        colname = root.split('\\')[-2].replace("col","")
        colnum = int(colname)
        model = models.load_model(modelpath)

        if os.path.exists(root + "\\result.csv"):
                os.remove(root + "\\result.csv")

        for file in files:

            if file.split('.')[-1] != "png":
                continue
            curImgPath = os.path.join(root , file)

            img = cv2.imread(curImgPath, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img , axis = -1)
            pred = Predict_Checkbox(img , colnum , model)

            

            with open( root + "\\result.csv" , 'a' , encoding = 'utf-8' ) as f:
                writer = csv.writer(f)
                writer.writerow([file , pred])


def main_failonly():
    failpath = r"F:\00_gitbio\OCR_bio\OCR_bio\data\checksplited\fail\col4\7_survey_16.png"
    modelpath = r'Check_Uncheck\CheckUncheckModel2.hdf5'
    model = models.load_model(modelpath)
    img = cv2.imread(failpath, cv2.IMREAD_GRAYSCALE)
    img = np.expand_dims(img , axis = -1)
    pred = Predict_Checkbox(img , 4 , model )

if __name__ == "__main__":
    #main_col5()
    main_col(5)
    #main_failonly()

