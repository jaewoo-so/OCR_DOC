
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

x = [1,2,4]
y = [2,4,8]

X = tf.placeholder( tf.float32 , None)
Y = tf.placeholder(tf.float32 , None)
W1 = tf.Variable( tf.random_uniform( [1] , -1.0 , 1.0 ) , dtype = tf.float32 ) 
b1 = tf.Variable( tf.random_uniform( [1] , -1.0, 1.0 ) , dtype = tf.float32  )


output = W1 * X + b1

cost_mse = tf.reduce_mean( tf.square( Y - output ) ) / 2.
optimizer = tf.train.GradientDescentOptimizer( 0.02 )
train_op = optimizer.minimize(cost_mse)

wList = []
bList = []
lossList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        sess.run([train_op] , feed_dict = {X : x , Y : y})
        cur_w , cur_b , cur_loss = sess.run( [W1 , b1 , cost_mse] , feed_dict = {X : x , Y : y} )
        wList.append(cur_w)
        bList.append(cur_b)
        lossList.append(cur_loss)
        
        print("w = {0}  , loss = {1}".format(cur_w, cur_loss))


plt.subplot(311)
plt.plot(lossList)
plt.title("loss")
plt.subplot(312)
plt.plot(wList)
plt.title("w")
plt.subplot(313)
plt.plot(bList)
plt.title("b")
plt.show()











