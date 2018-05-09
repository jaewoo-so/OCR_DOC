import os 
import shutil

dirpath = r"data\survey_result_Labeled"

n_section = 20
allPathList = None
infoList = None



for root , subdir , files in os.walk(dirpath):
    allPathList = [ os.path.join(root , filename) for filename in files ]
    infoList = [ list(x) for x in zip(allPathList , files)]
    

## split
def Splitter(allpathlist):
    dirpath = os.path.dirname(allpathlist[0])
    for path in allpathlist:
        name = os.path.basename(path)
        print(name)
        splited = name.split('_')
        sampleNb = splited[0]
        contentNb = splited[-1].split('.')[0]

        content_dir = os.path.join(dirpath,contentNb)

        if not os.path.exists(content_dir) :
            os.makedirs(content_dir)

        shutil.copy(path, os.path.join(content_dir , name))




Splitter(allPathList)








