import pandas as pd
import numpy as np 
import os
import csv 

testpath = r"data\hangul-images-data\labels-map.csv"
path = "test.csv"
#res = np.loadtxt( path ,dtype = str  )

res2 = pd.read_csv( testpath ,sep = ','  )

res3 = res2.iloc[: , 1]


print(res3.head())

res4 = list(res3.iloc[0:10])

print()




