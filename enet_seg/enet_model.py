from enet_seg.enet_blocks import *
import tensorflow as tf
import numpy as np

class ENet_Seg(tf.keras.Model):

    def section_2(self):

        layers_list = []

        # Bottleneck 2.1 Regular
        layers_list.append(RegularBN())

        # Bottleneck 2.2 Dilated 2
        layers_list.append(DilatedBN())

        # Bottleneck 2.3 Asymmetric 5
        layers_list.append(AsymmetricBN())

        # Bottleneck 2.4 Dilated 4
        layers_list.append(DilatedBN(dilation = 4))

        # Bottleneck 2.5 Regular
        layers_list.append(RegularBN())

        # Bottleneck 2.6 Dilated 8
        layers_list.append(DilatedBN(dilation = 8))

        # Bottleneck 2.7 Asymmetric 5
        layers_list.append(AsymmetricBN())

        # Bottleneck 2.8 Dilated 16
        layers_list.append(DilatedBN(dilation = 16))

        return layers_list

    def __init__(self, classes):

        super().__init__()

        layers_list = []

        # Initial block and Bottleneck 1.0
        self.initial_block = InitialBlock()
        self.bottleneck_1_0 = DownsampleBN(p_dropout = 0.01, ret_argmax = True, out_channel = 64)

        # 4 x Bottleneck 1.x
        for i in range(4):
            layers_list.append(RegularBN())

        
        # Bottleneck 2.0
        self.bottleneck_2_0 = DownsampleBN(ret_argmax = True, out_channel = 128)

        # Repeating section 2 twice
        layers_list.extend(self.section_2())
        layers_list.extend(self.section_2())

        self.intermediate = tf.keras.Sequential(layers_list)

        # Upsampling and Regular BN
        self.upsampling_1 = UpsampleBN(out_channel = 64)
        self.upsampling_2 = UpsampleBN(out_channel = 16)
        self.regular_1 = RegularBN()
        self.regular_2 = RegularBN()

        # Final Fully connected layer
        self.fully_conv = tf.keras.layers.Conv2DTranspose(filters = classes, kernel_size = (2, 2), strides = (2, 2))

    def call(self, image_input):

        initial_out = self.initial_block(image_input)
        bottle1_out, pool1_map = self.bottleneck_1_0(initial_out)
        bottle2_out, pool2_map = self.bottleneck_2_0(bottle1_out)

        # Blocks 2 and 3
        section_2_3 = self.intermediate(bottle2_out)

        # Block 4 : Upsampling and 2 Regular BNs
        upsample1_out = self.upsampling_1(section_2_3, pool2_map)
        regular_out = self.regular_1(upsample1_out)
        regular_out = self.regular_1(regular_out)

        # Block 5 : Upsampling and 1 Regular BN
        upsample2_out = self.upsampling_2(regular_out, pool1_map)
        regular_out = self.regular_2(upsample2_out)

        # Final conv
        output = self.fully_conv(regular_out)
        output = tf.keras.activations.softmax(output)
        return output














