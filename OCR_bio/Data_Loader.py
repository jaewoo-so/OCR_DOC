import pandas as pd
import csv
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import win_unicode_console
win_unicode_console.enable()

class HangulLoader:
   
    # with pandas, single data is excluded ...
    def GetImages_Labels(self,  imgpath,labelpath , isGray):
        #labels
        labels = pd.read_csv( labelpath , sep = ',' ,header=None, encoding = 'utf-8').iloc[:,1]
        list_labels = list(labels)
        set_labels = list(set(list_labels))
        one_hot_vectors = np.eye(len(set_labels), dtype=int)
        class_vectors = {}
        for i, cls in enumerate(set_labels):
            class_vectors[cls] = one_hot_vectors[i]
        yList = [ class_vectors[cate_name] for cate_name in list_labels]
        yList = np.asarray(yList)
        classnum = len(set_labels)
        #images 
        file_list = []
        for root,_,fnames in os.walk(imgpath):
            fnames.sort(key = (lambda x : int(x.split('_')[1].split('.')[0])))
            for filename in fnames:
                file_list.append(os.path.join(root, filename))
        imglist = np.asarray([ cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in file_list])

        if isGray:
            imglist = imglist.reshape(imglist.shape[0] , imglist.shape[1] , imglist.shape[2] , 1)
        imglist = imglist.astype('float32')
        imglist /= 255.0
        return imglist , yList , classnum

if __name__ == '__main__':
    testpath = r"data\hangul-images-data\labels-map.csv"
    dirpath = r"data\hangul-images-data\img"
    imgs , res = HangulLoader().GetImages_Labels(dirpath,testpath)

   
   






