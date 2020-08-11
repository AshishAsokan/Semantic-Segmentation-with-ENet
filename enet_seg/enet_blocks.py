import tensorflow as tf
from enet_seg.utilities import MaxUnpool2D
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

    def __init__(self, out_channel : int, filter_size : int = 3, pool_size : tuple = (2, 2), pool_stride : tuple = (2, 2), bn_pad : str = 'SAME', p_dropout : float = 0.1, ret_argmax : bool = False):

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
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.out_channel = out_channel
        self.bn_pad = bn_pad
        self.ret_argmax = ret_argmax

        # Maxpool and PReLU layers
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
        maxpool_output, argmax_pool = tf.nn.max_pool_with_argmax(block_input, ksize = self.pool_size, strides = self.pool_stride, padding = self.bn_pad)
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

        # Returning pooling indices for upsampling
        if self.ret_argmax:
            return (output, argmax_pool)

        return output

class RegularBN(tf.keras.layers.Layer):

    def __init__(self, filter_size : int = 3,  bn_pad : str = 'SAME', p_dropout : float = 0.1, filters : int = 4):

        """
        Regular Bottleneck block of the E-Net architecture.

        Parameters:

            bn_pad : Either 'SAME' or 'VALID' for pooling layer and conv in BRANCH-2
            out_channel : No of channels in output
            filter_size : Filter size for the conv operation in BRANCH-2
            p_dropout : Dropout rate of the regularizer

        Returns:

            output : Result of the element-wise addition of the 2 branch outputs with PReLU activation

        """

        super().__init__()
        self.filter_size = filter_size
        self.bn_pad = bn_pad
        self.filters = filters

        # PReLU layer
        self.b2_prelu = tf.keras.layers.PReLU()

        # Regularizer : Spatial Dropout with p = 0.01
        self.b2_reg = tf.keras.layers.SpatialDropout2D(rate = p_dropout, data_format = 'channels_last')

        # Batch normalization
        self.b2_batch_norm = tf.keras.layers.BatchNormalization()

    def build(self, input_shape):

        # Number of output channels
        channels_out = input_shape[3]

        # 1x1 convolution and conv of choice
        self.b2_conv1 = tf.keras.layers.Conv2D(filters = self.filters, kernel_size = (1, 1), strides = (1, 1))
        self.b2_conv2 = tf.keras.layers.Conv2D(filters = self.filters, kernel_size = (self.filter_size, self.filter_size), strides = (1, 1), padding = self.bn_pad)

        # 1x1 expansion layer after convolution
        self.b2_expand = tf.keras.layers.Conv2D(filters = channels_out, kernel_size = (1, 1), strides = (1, 1))

    def call(self, block_input):
        
        ######## BRANCH - 1 ###############

        # Storing input as Branch-1
        branch_1 = block_input

        ######## BRANCH - 2 ###############

        # 1x1 projection with BN and PReLU
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

        print(branch_2.shape)

        # Combining outputs of the 2 branches
        output = tf.keras.layers.Add()([branch_1, branch_2])
        output = tf.keras.layers.PReLU()(output)
        return output

class UpsampleBN(tf.keras.layers.Layer):

    def __init__(self, pool_argmax, out_channel : int, filter_size : int = 2, pool_size : tuple = (2, 2), bn_pad : str = 'SAME', p_dropout : float = 0.1):

        """
        Upsampling Bottleneck block of the E-Net architecture.

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

        # Reducing number of input channels
        self.b1_red = tf.keras.layers.Conv2D(filters = out_channel, kernel_size = (1, 1))
        self.b2_expand = tf.keras.layers.Conv2D(filters = out_channel, kernel_size = (1, 1))

        # Batch Normalisation and PReLU
        self.b2_batchnorm = tf.keras.layers.BatchNormalization()
        self.b2_prelu = tf.keras.layers.PReLU()

        # MaxUnpool layer
        self.b1_maxunpool = MaxUnpool2D(pool_mask = pool_argmax)

    def build(self, input_shape):

        # Feature map reduction in BRANCH-2
        filters = input_shape[3] // 4
        self.b2_red = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1, 1))

        # Deconvolution layer for BRAHCN-2
        filter_tup = (self.filter_size, self.filter_size)
        self.b2_deconv = tf.keras.layers.Conv2DTranspose(filters = filters, kernel_size = filter_tup, strides = (2, 2))


    def call(self, block_input):

        ######## BRANCH - 1 ###############

        # Reducing channels and MaxUnpool
        pool_out = self.b1_red(block_input)
        branch_1 = self.b1_maxunpool(pool_out)

        ######## BRANCH - 2 ###############

        # Reducing channels with 1x1 conv
        conv1_out = self.b2_red(block_input)
        batch_norm1 = self.b2_batchnorm(conv1_out)
        prelu1_out = tf.keras.layers.PReLU()(batch_norm1)
        
        # Deconvolution followed by BN and PReLU
        conv2_out = self.b2_deconv(prelu1_out)
        batch_norm2 = self.b2_batchnorm(conv2_out)
        prelu2_out = tf.keras.layers.PReLU()(batch_norm2)

        # 1x1 expansion
        conv3_out = self.b2_expand(prelu2_out)
        batch_norm3 = self.b2_batchnorm(conv2_out)
        branch_2 = tf.keras.layers.PReLU()(batch_norm3)

        # Adding branch outputs
        output = tf.keras.layers.Add()([branch_1, branch_2])
        output = tf.keras.layers.PReLU()(output)
        return output



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




    
