import tensorflow as tf
import numpy as np

class InitialBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters : int = 13, conv_kernel : tuple = (2, 2), conv_stride : tuple = (2, 2), padding : str = 'VALID',
        pool_kernel : tuple = (2, 2), pool_stride : tuple = (2, 2)):

        """
        Initial block of the E-Net architecture.

        Parameters:

            num_filters : No. of filters to be used for convolution = 13
            conv_kernel : Size of convolution filters
            conv_stride : Strides in both directions for convolution
            padding : Padding for maxpool operation (either 'SAME' or 'VALID')
            pool_kernel : Size of maxpool kernel
            pool_stride : Strides in both directions for maxpool operation

        Returns:

            final_output : 16 channel output formed by concatenating output of both branches
        """

        super().__init__()
        self.num_filters = num_filters
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.padding = padding

    def build(self, input_shape):

        # Convolution Layer, PRelu and BatchNorm
        self.convolution = tf.keras.layers.Conv2D(filters = self.num_filters, kernel_size = self.conv_kernel, strides = self.conv_stride)
        self.prelu = tf.keras.layers.PReLU()
        self.batch_norm = tf.keras.layers.BatchNormalization()

        # Maxpool Layer
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size = self.pool_kernel, strides = self.pool_stride, padding = self.padding)

    def call(self, block_input):
        
        con_output = self.convolution(block_input)
        batch_norm_output = self.batch_norm(con_output)
        rel_output = self.prelu(con_output)
        max_output = self.maxpool(block_input)

        final_output = tf.concat([con_output, max_output], axis = -1)
        return final_output

class DownsampleBN(tf.keras.layers.Layer):

    def __init__(self, out_channel : int, filter_size : int = 3, pool_size : tuple = (2, 2), pool_stride : tuple = (2, 2), bn_pad : str = 'SAME'):

        """
        Downsampling Bottleneck block of the E-Net architecture.

        Parameters:

            pool_size : Tuple for pooling filter size
            pool_stride : Strides in both directions for pooling layer
            bn_pad : Either 'SAME' or 'VALID' for pooling layer and conv in BRANCH-2
            in_channel : No of channels in input
            out_channel : No of channels in output
            filter_size : Filter size for the conv operation in BRANCH-2

        Returns:

            output : Result of the element-wise addition of the 2 branch outputs with PReLU activation

        """

        super().__init__()
        self.filter_size = filter_size
        self.out_channel = out_channel
        self.bn_pad = bn_pad

        # Maxpool and PReLU layers
        self.b1_maxpool = tf.keras.layers.MaxPool2D(pool_size = pool_size, strides = pool_stride, padding = bn_pad)
        self.b2_prelu = tf.keras.layers.PReLU()
        
        # 1x1 expansion layer after convolution
        self.b2_expand = tf.keras.layers.Conv2D(filters = out_channel, kernel_size = (1, 1), strides = (1, 1))

        # Regularizer : Spatial Dropout with p = 0.01
        self.b2_reg = tf.keras.layers.SpatialDropout2D(rate = 0.01, data_format = 'channels_last')

        # Batch normalization
        self.b2_batch_norm = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):

        # Calculating the number of channels to pad
        in_channel = input_shape[3]
        pad = abs(in_channel - self.out_channel)
        self.pad_tensor = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, pad]], dtype = tf.int32)

        # No. of filters for convolution
        filters = self.out_channel // in_channel

        # 2x2 convolution and conv of choice
        self.b2_conv1 = tf.keras.layers.Conv2D(filters = filters, kernel_size = (2, 2), strides = (2, 2))
        self.b2_conv2 = tf.keras.layers.Conv2D(filters = filters, kernel_size = (self.filter_size, self.filter_size), strides = (1, 1), padding = self.bn_pad)

    def call(self, block_input):
        
        ######## BRANCH - 1 ###############

        # Maxpool followed by padding the channels
        maxpool_output = self.b1_maxpool(block_input)
        branch_1 = tf.pad(maxpool_output, paddings = self.pad_tensor)

        ######## BRANCH - 2 ###############

        # 1st convolution with BN and PReLU
        conv1_out = self.b2_conv1(block_input)
        batch1_out = self.b2_batch_norm(conv1_out)
        prelu1_out = self.b2_prelu(batch1_out)

        # 2nd convolution with BN and PReLU
        conv2_out = self.b2_conv2(prelu1_out)
        batch2_out = self.b2_batch_norm(conv2_out)
        prelu2_out = self.b2_prelu(batch2_out)

        # 1x1 expansion and spatial dropout
        exp1_out = self.b2_expand(prelu2_out)
        branch_2 = self.b2_reg(exp1_out)

        # Combining outputs of the 2 branches
        output = tf.keras.layers.Add()([branch_1, branch_2])
        output = tf.keras.layers.PReLU()(output)
        return output


class UpsampleBN(tf.keras.Model):

    def __init__(self):

        super().__init__()

    def call():
        raise NotImplementedError

class AntisymmetricBN(tf.keras.Model):
    def __init__(self):

        super().__init__()

    def call():
        raise NotImplementedError

class DilatedBN(tf.keras.Model):

    def __init__(self):

        super().__init__()

    def call():
        raise NotImplementedError

class RegularBN(tf.keras.Model):

    def __init__(self):

        super().__init__()

    def call():
        raise NotImplementedError




    
