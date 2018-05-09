from keras.layers import *
from keras import losses , models ,activations , optimizers
from keras import backend as K
import tensorflow as tf

class CreateC1:

    def moedeltest():
        input = Input((4))
        h1 = Dense(6 , activation = "sigmoid")(input)
        h2 = Dense(2 , activation = "sigmoid ")(h1)

        #####
        x = tf.placeholder( tf.float32 , (None , 4))
        y = tf.placeholder(tf.float32 , (None , 2))

        W1 = tf.Variable( tf.ranœœdom_uniform( (4,6) , -1.0 , 1.0 ) , dtype = tf.float32 ) 
        b1 = tf.Variable( tf.random_uniform( (4) , -1.0, 1.0 ) , dtype = tf.float32 )

        l1_input = tf.matmul( x , W1 ) + b1
        L1 = tf.nn.sigmoid( l1_input )
        ####



        model = models.Model(input,h2)

        model.compile( loss = losses.binary_crossentropy , optimizer = optimizers.Adam(0.02) )

        history = model.fit(x,y , batch_size  =  64 , validation_split = 0.2 , shuffle = True )

        pred = model.predict()


    def CNNBlock(self,n_out,input):
        x = Conv2D(n_out,(3,3) ,activation = 'relu')(input)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Conv2D(n_out,(3,3), activation = 'relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = MaxPool2D((2,2))(x)
        return x


    def GetCNNModel(self, input):
        
        l1 = self.CNNBlock(64 , input)
        l2 = self.CNNBlock(128 , l1)
        #l3 = self.CNNBlock(128 , l2)
        #l4 = self.CNNBlock(256 , l3)
        #l5 = self.CNNBlock(512 , l4)
       
        return l2


