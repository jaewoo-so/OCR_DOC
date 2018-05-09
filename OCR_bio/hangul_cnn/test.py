import numpy as np
import tensorflow as tf

x = [[1],[2],[4]]
y = [2,4,8]


X = tf.placeholder( tf.float32 , (3,1))
Y = tf.placeholder(tf.float32 , (3))
W1 = tf.Variable( tf.random_uniform( (1) , -1.0 , 1.0 ) , dtype = tf.float32 ) 
b1 = tf.Variable( tf.random_uniform( (1) , -1.0, 1.0 ) , dtype = tf.float32  )

####
output = X * W1  + b1
####

cost_mse = tf.reduce_mean( tf.square( Y - output ) ) / 2.

optimizer       = tf.train.GradientDescentOptimizer( 0.2 )

train_op = optimizer.minimize(cost_mse)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(3):
        sess.run([train_op] , feed_dict = {X : x , Y : y})
        cur_w , cur_loss = sess.run( [W1 , cost_mse] , feed_dict = {X : x , Y : y} )

        print()













