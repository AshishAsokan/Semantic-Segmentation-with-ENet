import tensorflow as tf
from enet_seg.enet_blocks import *
from enet_seg.enet_model import ENet_Seg
from enet_seg.utilities import MaxUnpool2D
import pytest

def test_InitialBlock():

    """ A simple test that checks the output of the Initial Block """

    # Initializing random input
    tf.random.set_seed(1)
    test_input = tf.random.uniform(shape=[4, 512, 512, 3])

    # Creating the layer and appling InitialBlock
    initial_block = InitialBlock()
    output = initial_block(test_input)

    assert output.shape == (4, 256, 256, 16)

def test_DownsampleBN():

    """ A simple test that checks the output of the Initial Block """

    # Initializing random input
    tf.random.set_seed(1)
    test_input = tf.random.uniform(shape=[5, 256, 256, 16])

    # Creating the layer and appling Downsample
    downsample_block = DownsampleBN(64)
    output = downsample_block(test_input)

    assert output.shape == (5, 128, 128, 64)

def test_MaxUnpool2D():

    """ A simple test that checks the output of the MaxUnpool Layer """

    # Initializing random input
    tf.random.set_seed(1)
    test_input = tf.random.uniform(shape=[5, 128, 128, 64])

    # Performing maxpool with argmax
    pool_output, pool_mask = tf.nn.max_pool_with_argmax(test_input, ksize = (2, 2), strides = (2, 2), padding = 'VALID')

    # Creating the layer and appling Regular BN
    maxunpool = MaxUnpool2D(pool_mask = pool_mask)
    output = maxunpool(pool_output)

    assert output.shape == (5, 128, 128, 64)

def test_RegularBN():

    """ A simple test that checks the output of the Regular Bottleneck """

    # Initializing random input
    tf.random.set_seed(1)
    test_input = tf.random.uniform(shape=[5, 128, 128, 64])

    # Creating the layer and appling Regular BN
    regular_bn = RegularBN()
    output = regular_bn(test_input)

    assert output.shape == (5, 128, 128, 64)

def test_UpsampleBN():

    """ A simple test that checks the output of the Unsample Bottleneck """

    # Initializing random input
    tf.random.set_seed(1)
    test_input = tf.random.uniform(shape=[5, 256, 256, 16])

    # Applying Maxpool
    downsample = DownsampleBN(64, ret_argmax = True)
    (down_out, pool_mask) = downsample(test_input)

    # Creating the layer and appling Upsample BN
    upsample_bn = UpsampleBN(out_channel = 16)
    output = upsample_bn(down_out, pool_mask)

    assert output.shape == (5, 256, 256, 16)

def test_AsymmtericBN():

    """ A simple test that checks the output of the Antisymmetric Bottleneck """

    # Initializing random input
    tf.random.set_seed(1)
    test_input = tf.random.uniform(shape=[5, 128, 128, 64])

    # Creating the layer and appling Regular BN
    asym_bn = AsymmetricBN()
    output = asym_bn(test_input)

    assert output.shape == (5, 128, 128, 64)

def test_DilatedBN():

    """ A simple test that checks the output of the Dilated Bottleneck """

    # Initializing random input
    tf.random.set_seed(1)
    test_input = tf.random.uniform(shape=[5, 128, 128, 64])

    # Creating the layer and appling Regular BN
    dilated_bn = DilatedBN()
    output = dilated_bn(test_input)

    assert output.shape == (5, 128, 128, 64)

def test_ENetModel():

    """ A simple test to check the output of the ENet Model """

    # Initializing random input
    tf.random.set_seed(0)
    input_data = tf.random.uniform(shape = [5, 512, 512, 3])

    # Creating Enet Model with 16 classes
    model = ENet_Seg(classes = 16)
    output = model(input_data)
    
    assert output.shape == (5, 512, 512, 16)

if __name__ == '__main__':
    pytest.main([__file__])

