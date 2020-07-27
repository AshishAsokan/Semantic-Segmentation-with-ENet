import tensorflow as tf
import numpy as np

class EnetInitialBlock(tf.keras.Model):

    def __init__(self, num_filters : int, kernel_sizes : tuple, stride : tuple, padding):

        super.__init__()

        # Defining the convolution and maxpool layers
        self.convolution = tf.keras.layers.Conv2D(filters = num_filters, kernel_size = kernel_sizes[0], stride = strides[0])
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size = kernel_sizes[1], strides = stride[0], padding = padding)

    def call(self, image):
        
        """
        Overloading the call function for the Model Class

        Parameters:
            image : Input image for the ENet Block

        Returns:
            final_output : Final concatenated output with 16 channels
        """
        con_output = self.convolution(image)
        max_output = self.maxpool(image)

        final_output = tf.concat([con_output, max_output], axis = -1)
        return final_output

class BottleneckBlock(tf.keras.Model):

    def __init__(self, downsample : bool, dilated : int, assymetric : int, upsample : bool, rate : float):

        super.__init__()

        
        ### BRANCH - 1
        if upsample:
            self.maxnpool = tf.keras.layers.MaxPool2D(pool_size = )

        ### BRANCH - 2

        # Convolution is 2x2 for downsampling bottleneck
        if downsample:
            self.conv_1 = tf.keras.layers.Conv2D(filters = 1, kernel_size = (2, 2), stride = (2, 2), activation = None)
        else:
            self.conv_1 = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1, 1), stride = (1, 1), activation = None)

        # Defining the PReLU layer and regularizer
        self.PReLU = tf.keras.layers.PReLU(alpha_initializer = 'zeros')
        self.reg = tf.keras.layers.Dropout(rate = rate)

        self.conv_2 = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1, 1), stride = (1, 1), activation = None)









def bottleneck(bottle_input, downsample = False, dilated = 0, assymetric = 0, upsample = False):

    conv_1 = None

    # Defining the bottleneck layers
    if not downsample:
        conv_1 = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1, 1), stride = (1, 1), activation = None)
    else:
        conv_1 = tf.keras.layers.Conv2D(filters = 1, kernel_size = (2, 2), stride = (2, 2), activation = None)


    prelu_1 = tf.keras.layers.PReLU(alpha_initializer = 'zeros')
    regularizer = tf.keras.layers.Dropout(rate = 0.1)




    
