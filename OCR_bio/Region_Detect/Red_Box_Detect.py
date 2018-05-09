import cv2


'''

class Architecture 

IO
 |
Applicaation 
 |
Function Class , Data Class

***All function is pure function

'''



# This class is for method metrial. base level 
# you have to write other class that perform high level api with these functions 
class DocAutoReader_FnLib:

    def _BGR2HSV(self , img):

        img_hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)



        pass

    ## in opencv , HSV range is 0 to 255
    def getOnlRredArea(self,img , hsvmin, hsvmax):


        pass 

    def getRectRegion(self , img , min , max):

        _ , contours , _ = cv2.findContours( img , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
        rectList = []
        for contour in contours:
            rect = Rectangle( cv2.boundingRect( contours ) )

            if rect.area < max and rect.area > min:
                rectList.append( rect )

        return rectList 

    def getRoiImgFromRectList(self, img , rectList):
        roiList = [ rect.CropFrom(img) for rect in rectList ]
        return roiList


class Rectangle:
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = w*h

    def CropFrom(self,img):
        return img[ self.y : self.y + self.h , self.x : self.x+self.w ]


# All information of document for vision processing need to be stored here
# All method for processing use configData class for recipe
class ConfigData:

    def __init__(self):

        self.Survey_1_max = 210000 
        self.Survey_1_min = 180000 

        self.Survey_2_max = 420000 
        self.Survey_2_min = 340000 

        self.Survey_3_max = 440000 
        self.Survey_3_min = 400000 

        self.Survey_4_max = 160000 
        self.Survey_4_min = 140000 

        self.Survey_5_max = 114000 
        self.Survey_5_min = 113000 

        self.Survey_6_max = 112900 
        self.Survey_6_min = 110000 

# this class is for data provider for test 
# have to put config parameter to 
class TestConfig:

    def __init__(self, configDataClass):
        self.config = configDataClass

    def Survey_1(self):
        return self.config.Survey_1_max , self.config.Survey_1_min

    def Survey_2(self):
        return self.config.Survey_2_max , self.config.Survey_2_min

    def Survey_3(self):
        return self.config.Survey_3_max , self.config.Survey_3_min

    def Survey_4(self):
        return self.config.Survey_4_max , self.config.Survey_4_min

    def Survey_5(self):
        return self.config.Survey_5_max , self.config.Survey_5_min

    def Survey_6(self):
        return self.config.Survey_6_max , self.config.Survey_6_min



