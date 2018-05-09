import numpy as np
from matplotlib import pyplot as plt

###
class TestDataLoader:
    def f1(self,x):
        y = 3 * x + 4
        return y + np.random.random()*5
    
    def f2(self,x):
        y = 3 * x  - 4 
        return y + np.random.random()*5
    
    def GetTestData(self):
        x1=  np.arange(0,5,0.2)
        x2=  np.arange(5,10,0.2)
        y1 = np.asarray([ self.f1(x) for x in x1 ] )
        y2 = np.asarray([ self.f2(x) for x in x2 ] )
        return x1,x2,y1,y2

    def ShowData(self):
        x1,x2,y1,y2 = self.GetTestData()

        xdata = x1 + x2
        ydata = y1 + y2 
        
        plt.scatter(x1,y1 , color = 'r')
        plt.scatter(x2,y2 , color = 'g')
        plt.show()
    
if __name__ == '__main__':
    dl = TestDataLoader()
    dl.ShowData()








