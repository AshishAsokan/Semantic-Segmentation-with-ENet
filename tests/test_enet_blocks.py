import tensorflow as tf
from enet_seg.enet_blocks import InitialBlock, DownsampleBN
from enet_seg.utilities import MaxUnpool2D
import pytest

def test_InitialBlock():

    """ A simple test that checks the output of the Initial Block """

    # Initializing random input
    tf.random.set_seed(1)
    test_input = tf.random.uniform(shape=[4, 512, 512, 3])

    # Creating the layer and appling FFM
    initial_block = InitialBlock()
    output = initial_block(test_input)

    assert output.shape == (4, 256, 256, 16)

def test_DownsampleBN():

    """ A simple test that checks the output of the Initial Block """

    # Initializing random input
    tf.random.set_seed(1)
    test_input = tf.random.uniform(shape=[5, 256, 256, 16])

    # Creating the layer and appling FFM
    downsample_block = DownsampleBN(64)
    output = downsample_block(test_input)

    assert output.shape == (5, 128, 128, 64)

def test_MaxUnpool2D():

    """ A simple test that checks the output of the MaxUnpool Layer """

    # Initializing random input
    tf.random.set_seed(1)
    test_input = tf.random.uniform(shape=[5, 30, 30, 30])

    # Performing maxpool with argmax
    pool_output, pool_mask = tf.nn.max_pool_with_argmax(test_input, ksize = (2, 2), strides = (2, 2), padding = 'VALID')

    # Creating the layer and appling FFM
    maxunpool = MaxUnpool2D(pool_mask = pool_mask)
    output = maxunpool(pool_output)

    assert output.shape == (5, 30, 30, 30)


if __name__ == '__main__':
    pytest.main([__file__])

