
import numpy as np
import tensorflow as tf
from TutorialDataLoader import TestDataLoader as DL
from matplotlib import pyplot as plt
dl = DL()
#dl.ShowData()

x1,x2,y1,y2 = dl.GetTestData()

x = x1 + x2
y = y1 + y2

labels = np.hstack( (np.zeros( len(x1) , dtype = np.int32 ) , np.ones( len(x2), dtype = np.int32 )))

laels = np.eye(2)[ labels.reshape(-1) ]
xdata = np.dstack((x,y))[0]

print(xdata.shape)
print(labels.shape)
print()




def WithFirstModel(x,y):
    X = tf.placeholder( tf.float32 , (None,2))
    Y = tf.placeholder(tf.float32 , (None,2))
    
    W1 = tf.Variable( tf.random_uniform( [2,2] , -1.0 , 1.0 ) , dtype = tf.float32 ) 
    b1 = tf.Variable( tf.random_uniform( [2] , -1.0, 1.0 ) , dtype = tf.float32  )
    
    output = tf.matmul(X , W1)  + b1
    
    cost_mse = tf.reduce_mean( tf.square( Y - output ) ) / 2.
    optimizer = tf.train.GradientDescentOptimizer( 0.2 )
    train_op = optimizer.minimize(cost_mse)
    
    lossList = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        for step in range(3):
            sess.run([train_op] , feed_dict = {X : x , Y : y})
            cur_loss = sess.run( [ cost_mse] , feed_dict = {X : x , Y : y} )
            lossList.append(cur_loss)
            print("loss = {1}".format(cur_w, cur_loss))

        plt.plot(lossList)
        plt.title("loss")
        plt.show()

        pred = sess.run([output] , feed_dict = {X : x , Y : y})
        
        for o , t in zip(pred , y):
            print("Pred : {0}  ,  Target : {1}".format(o , t))


def WithSecondModel(x,y):
    pass


WithFirstModel(xdata,labels)

WithSecondModel(xdata,labels)






