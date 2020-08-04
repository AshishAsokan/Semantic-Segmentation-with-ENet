import tensorflow as tf
import os
from enet_seg.enet_blocks import InitialBlock, DownsampleBN
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


if __name__ == '__main__':
    pytest.main([__file__])

