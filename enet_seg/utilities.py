import tensorflow as tf
import numpy as np

class MaxUnpool2D(tf.keras.layers.Layer):

    """
    Implementation of the MaxUnpool layer outlined in the original ResNet paper.
    This is based on https://github.com/kwotsin/TensorFlow-ENet.git
    """

    def __init__(self, kernel_size: list, pool_mask):

        super().__init__()
        self.kernel_size = kernel_size
        self.pool_mask = tf.cast(pool_mask, tf.int32)

    def build(self, shape_input):

        # Defining shape of output
        self.shape_output = [shape_input[0], shape_input[1] * self.kernel_size[1],
                             shape_input[2] * self.kernel_size[2], shape_input[3]]

    def call(self, input_pool):

        # For the first channel : No of batches
        ones_mask = tf.ones_like(pool_mask, dtype=tf.int32)

        # Output shape with no of batches
        batch_shape = (shape_output[0], 1, 1, 1)
        batch_range = tf.reshape(tf.range(shape_output[0], dtype=tf.int32), shape=batch_shape)
        batch_size = tf.cast(ones_mask * batch_range, dtype=tf.int32)

        # For x and y (Middle 2) channels
        channel_2 = tf.cast(pool_mask // (shape_output[2] * shape_output[3]), dtype=tf.int32)
        channel_3 = tf.cast((pool_mask // shape_output[3]) % shape_output[2], dtype=tf.int32)

        # For the last channel (Number of feature maps)
        feature_mask = tf.range(shape_output[3], dtype=tf.int32)
        features = ones_mask * feature_mask

        # Indices for merging with scatter_nd
        input_pool_size = tf.size(output)
        indices = tf.transpose(tf.reshape(tf.stack([batch_size, channel_2, channel_3, features]), [4, input_pool_size]))
        values = tf.reshape(output, [input_pool_size])

        # Using scatter to merge for the final final output
        max_unpool = tf.scatter_nd(indices, values, shape_output)
        return max_unpool
