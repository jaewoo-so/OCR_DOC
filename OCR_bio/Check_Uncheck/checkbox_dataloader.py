import os 
import cv2 
import numpy as np

class FileInfo:
    def __init__(self, sampleNum , fullPath , contentNum , imgsizeWH = (240, 38) ):
        self.sample_n = sampleNum
        self.fullpath = fullPath
        self.contetnNum = contentNum
        self.imgSize = imgsizeWH
        self.resizedImg = None
        self.label = None
        
        self.SetResizedImg()

    def SetResizedImg(self):
        img = cv2.imread( self.fullpath , cv2.IMREAD_GRAYSCALE ).astype('float32')
        img = cv2.resize( img , self.imgSize )
        img = img/255.
        img = np.expand_dims(img, axis = 2)

        self.resizedImg = img


def SimpleLoader(path):
    imgList = []
    for root , _ , files in os.walk(path):
        for file in files:
            fileName = os.path.join(root, file)
            img = cv2.imread(fileName , cv2.IMREAD_GRAYSCALE)
            imgList.append(img)

    return imgList

def SimpleLoader_FixedSize(path,sizewh):
    imgList = []
    for root , _ , files in os.walk(path):
        for file in files:
            fileName = os.path.join(root, file)
            img = cv2.imread(fileName , cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img , sizewh )
            imgList.append(img)

    return imgList


def LoadCol5(basePath ):
    infoList = []
    labels = []
    for root , _ , files in os.walk(basePath):
        if len(files) == 0:
            continue
        cur_infolist = []
        currentLabels = []

        for filename in files :
            filename = os.path.join(root,filename)
            _ , ext = os.path.splitext(filename)

            if  filename.split('\\')[-1] == "result.csv":
                continue

            if  filename.split('\\')[-1] == "result.csv":
                continue

            if  ext == ".csv" :
                currentLabels = np.loadtxt(filename , dtype = np.int32, delimiter = ',')
                continue
                
            splited = filename.split('_')
            sam_nb = splited[-2].split('\\')[-1]
            content_nb = splited[-1].split('.')[0]
            fullpath = os.path.join( root, filename )
            cur_infolist.append( FileInfo( sam_nb , filename , content_nb ) )

    sorted( infoList , key = lambda x : x.sample_n )
    for i in range(len(currentLabels)):
        cur_infolist[i].label = currentLabels[i]
    infoList = infoList + cur_infolist 

    ys = np.asarray([ int(info.label) for info in infoList ] , dtype = np.int32)
    xs = np.asarray([ info.resizedImg for info in infoList ])

    ys = np.eye( 5 , dtype = np.int32 )[ ys - 1 ]

    # (9, 38, 240, 1)  = xs.shape for one loop //

    return xs , ys , (infoList[0].imgSize[1],infoList[0].imgSize[0])

def LoadCol(basePath , colNum ):
    infoList = []
    labels = []
    for root , _ , files in os.walk(basePath):
        if len(files) == 0:
            continue
        cur_infolist = []
        currentLabels = []

        for filename in files :
            filename = os.path.join(root,filename)
            _ , ext = os.path.splitext(filename)

            if  filename.split('\\')[-1] == "result.csv":
                continue

            if  ext == ".csv" :
                currentLabels = np.loadtxt(filename , dtype = np.int32, delimiter = ',')
                continue
                
            splited = filename.split('_')
            sam_nb = splited[0]
            content_nb = splited[-1].split('.')[0]
            fullpath = os.path.join( root, filename )
            cur_infolist.append( FileInfo( sam_nb , filename , content_nb ) )
       
    sorted( infoList , key = lambda x : x.sample_n )
    
    for i in range(len(currentLabels)):
        cur_infolist[i].label = currentLabels[i]
    
    infoList = infoList + cur_infolist 

    ys = np.asarray([ int(info.label) for info in infoList ] , dtype = np.int32)
    xs = np.asarray([ info.resizedImg for info in infoList ])

    ys = np.eye( colNum , dtype = np.int32 )[ ys - 1 ]
    return xs , ys , (infoList[0].imgSize[1],infoList[0].imgSize[0])



def DataSpliter(imglist , labellist , colNum , inputImgsize):

    sepimglist = []
    seplabellist = []
    gstep = None

    for  i , img in enumerate(imglist):
        h , w , c = img.shape
        step = int(w / colNum)
        gstep = step
        splitedimg = []
        labels = []
        for k in range(colNum):
            start = k*step
            end = (k+1)*step

            splitedimg.append( extractMargin(img[:,start : end , :],1))
            labels.append( labellist[i][k])

        sepimglist.append( splitedimg )
        seplabellist.append(labels)
    return sepimglist , seplabellist

# unroll all 5d data
def splitUnroll(nxs,nys):
    nxs = np.asarray(nxs)
    nys = np.asarray(nys)
    flatxs = nxs.reshape( (-1 , nxs.shape[-3], nxs.shape[-2] ,nxs.shape[-1] ) ) 
    flatys = nys.reshape( (-1 ) ) 
    return flatxs , flatys

def extractMargin(img,margin):
    h , w , c = None , None , None
    if len(img.shape) == 3 :
        h , w , c= img.shape
    else: 
        h , w = img.shape

    xstart = margin
    xend = w-margin
    ystart = margin
    yend = h - margin
    img = img[ystart:yend , xstart : xend]
    return img

# move file to each label dir
def splitAndSave(xs,ys,basepath):
    if not os.path.exists(basepath):
        os.mkdir(basepath)

    class_set = set(ys.tolist())
    class_dir = {}
    for cls in class_set:
        clsPath = os.path.join( basepath , str(cls) )
        if not os.path.exists(clsPath):
            os.mkdir( clsPath )
        class_dir[cls] = clsPath

    i = 0
    for x,y in zip(xs,ys):
        new_x = np.asarray( x * 255 , dtype = np.uint8)
        _ , new_x = cv2.threshold( new_x , 220  , 255 , cv2.THRESH_BINARY_INV )
        print(new_x.shape)
        savepath = os.path.join( class_dir[y] , str(i) + ".png" )
        new_x = extractMargin(new_x, 7 )
        cv2.imwrite( savepath , new_x)
        i += 1

def saveImg(flatxs):
    i = 0
    for i , img in enumerate( flatxs):
        img = img * 255
        cv2.imwrite( os.path.join( 'data\\test' , str(i) + '_' + str(flatys[i]) + ".jpg" ) ,img )
        i += 1

def Generate_EachColumn_Splited_Image(dirpath , colNum):
    # Load Process
    xs , labels ,  size = LoadCol(dirpath , colNum)
    nxs , nys = DataSpliter(xs,labels,colNum,xs[0].shape)
    flatxs , flatys = splitUnroll(nxs,nys)
    basepath = r"data\checksplited"

    # This Op , Threshold image and split only 0 , 1 
    splitAndSave(flatxs, flatys , basepath)

def main_col5():
    col5path = r'F:\00_gitbio\OCR_bio\OCR_bio\data\survey_result_Labeled\col5'
    # Load Process
    xs , labels ,  size = LoadCol5(col5path)
    nxs , nys = DataSpliter(xs,labels,5,xs[0].shape)
    flatxs , flatys = splitUnroll(nxs,nys)
    basepath = r"data\checksplited"

    # This Op , Threshold image and split only 0 , 1 
    splitAndSave(flatxs, flatys , basepath)
   
def main_col2():
    col5path = r'F:\00_gitbio\OCR_bio\OCR_bio\data\survey_result_Labeled\col2'
    # Load Process
    xs , labels ,  size = LoadCol(col5path,2)
    nxs , nys = DataSpliter(xs,labels,5,xs[0].shape)
    flatxs , flatys = splitUnroll(nxs,nys)
    basepath = r"data\checksplited"

    # This Op , Threshold image and split only 0 , 1 
    splitAndSave(flatxs, flatys , basepath)


if __name__ == "__main__":
    col2path = r'F:\00_gitbio\OCR_bio\OCR_bio\data\survey_result_Labeled\col2'
    col4path = r'F:\00_gitbio\OCR_bio\OCR_bio\data\survey_result_Labeled\col4'

    Generate_EachColumn_Splited_Image(col2path , 2)
