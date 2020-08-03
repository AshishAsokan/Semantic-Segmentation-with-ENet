import tensorflow as tf
import numpy as np

class InitialBlock(tf.keras.Model):

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


class UpsampleBN(tf.keras.Model):

    def __init__(self):

        super().__init__()

    def build(self):
        raise NotImplementedError

    def call():
        raise NotImplementedError

class DownsampleBN(tf.keras.Model):

    def __init__(self):

        super().__init__()

    def build(self):
        raise NotImplementedError

    def call():
        raise NotImplementedError

class AntisymmetricBN(tf.keras.Model):
    def __init__(self):

        super().__init__()

    def build(self):
        raise NotImplementedError

    def call():
        raise NotImplementedError

class DilatedBN(tf.keras.Model):

    def __init__(self):

        super().__init__()

    def build(self):
        raise NotImplementedError

    def call():
        raise NotImplementedError

class RegularBN(tf.keras.Model):

    def __init__(self):

        super().__init__()

    def build(self):
        raise NotImplementedError

    def call():
        raise NotImplementedError




    
