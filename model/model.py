from keras import Model
from keras.layers import (Input, Conv2D, Dropout,
                          AveragePooling2D,
                          Flatten, Dense, Activation, Add,
                          Concatenate, BatchNormalization)
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.utils import print_summary

class RedshiftClassifierResNet(Model):
    def __init__(self,
                 input_img_shape,
                 num_redshift_classes,
                 num_res_blocks=6,
                 num_res_stacks=4,
                 init_num_filters=16):
        """Initializes the ResNet model

           num_res_blocks refers to the number of residual blocks per stack,
           and the current version includes three stacks. In this context,
           a stack is a contiguous stream of layers in which all layers have
           the same number of filters and the same input image dimensions. Each
           time a new stack is entered, the number of filters doubles and the
           image dimensions are halved along the x and y axes.

           Given num_res_blocks = n and num_res_stacks = k, the actual depth of
           the network is 2kn + 3 trainable layers. By default, this
           network is a ResNet51 (identical to standard ResNet50 with an extra
           dense layer before the final output).

           Due to the image downsampling each stack, num_res_stacks should be
           in the range 1 <= k <= log2(num_res_stacks - 2) 
        """
        self.num_classes = num_redshift_classes

        # Input Layer Galactic Images
        image_input = Input(shape=input_img_shape)
        
        num_filters = init_num_filters

        # Initial Convolution Layer
        conv_init = Conv2D(num_filters,
                        kernel_size=(3,3),
                        strides=1,
                        padding='same',
                        activation='relu')
        weights = BatchNormalization()(conv_init(image_input))

        for stack in range(num_res_stacks):
            for block in range(num_res_blocks):
                weights = self.add_residual_block(weights,
                                                  num_filters=num_filters,
                                                  ds_input=(stack>0 and block==0))
            num_filters *= 2

        #* Original paper uses GlobalAveragePooling in lieu of avg pool + dense,
        #* may be worth implementing and comparing

        # End Pooling Layer
        pooling_layer = AveragePooling2D(pool_size=int(weights.shape[1]))
        pooling_layer_out = pooling_layer(weights)

        # Fully Connected Layers
        input_to_dense = Flatten(data_format='channels_last')(pooling_layer_out)
        model_output = Dense(units=num_redshift_classes, activation='softmax')(
                Dense(units=1024, activation='relu')(input_to_dense))
    
        super().__init__(inputs=[image_input],outputs=model_output)
        self.compile(optimizer=Adam(lr=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['sparse_categorical_accuracy'])

    def add_residual_block(self,
                           input_weights,
                           num_filters=16,
                           ds_input=False):
        """Adds a block composed of multiple residual layers"""

        #reslayer1
        conv1 = Conv2D(num_filters,
                       kernel_size=3,
                       strides=2 if ds_input else 1,
                       padding='same')
        conv1_out = conv1(Activation('relu')(BatchNormalization()(input_weights)))

        #reslayer2
        conv2 = Conv2D(num_filters,
                       kernel_size=3,
                       strides=1,
                       padding='same')
        conv2_out = conv2(BatchNormalization()(conv1_out))

        #skip connect
        if ds_input:
            # linearly downsample the input to allow addition with downsampled
            # convolved outputs. Generally should only happen in the first block
            # of each stack
            res_conv = Conv2D(num_filters,
                              kernel_size=1,
                              strides=2,
                              padding='same')
            res_weights = res_conv(input_weights)
        else:
            res_weights = input_weights

        output_weights = Activation('relu')(Add()([conv2_out,res_weights]))

        return output_weights

    def __str__(self):
        self.summary()
        return ""

    def __repr__(self):
        return f"RedshiftResNet({self.input_shape[1:]},{self.output_shape[1]})"

class RedshiftClassifierInception(Model):
    """
        This class is adapted from the model written by Umesh Timalsina
        of the Institute for Software Integrated Systems. The original
        code was pulled from https://github.com/umesh-timalsina/redshift/blob/master/model/model.py
        on 2/5/2020
    """
    def __init__(self,
                 input_img_shape,
                 num_redshift_classes):
        """Initialize the model"""
        self.num_classes = num_redshift_classes

        # Input Layer Galactic Images
        image_input = Input(shape=input_img_shape)
        # Convolution Layer 1
        conv_1 = Conv2D(64,
                        kernel_size=(5, 5),
                        padding='same',
                        activation='relu')
        conv_1_out = conv_1(image_input)

        # Pooling Layer 1
        pooling_layer1 = AveragePooling2D(pool_size=(2, 2),
                                          strides=2,
                                          padding='same')
        pooling_layer1_out = pooling_layer1(conv_1_out)

        # Inception Layer 1
        inception_layer1_out = self.add_inception_layer(pooling_layer1_out,
                                                        num_f1=48,
                                                        num_f2=64)

        # Inception Layer 2
        inception_layer2_out = self.add_inception_layer(inception_layer1_out,
                                                        num_f1=64,
                                                        num_f2=92)

        # Pooling Layer 2
        pooling_layer2 = AveragePooling2D(pool_size=(2, 2),
                                          strides=2,
                                          padding='same')
        pooling_layer2_out = pooling_layer2(inception_layer2_out)

        # Inception Layer 3
        inception_layer3_out = self.add_inception_layer(pooling_layer2_out, 92, 128)

        # Inception Layer 4
        inception_layer4_out = self.add_inception_layer(inception_layer3_out, 92, 128)

        # Pooling Layer 3
        pooling_layer3 = AveragePooling2D(pool_size=(2, 2),
                                          strides=2,
                                          padding='same')
        pooling_layer3_out = pooling_layer3(inception_layer4_out)

        # Inception Layer 5
        inception_layer5_out = self.add_inception_layer(pooling_layer3_out,
                                                        92, 128,
                                                        kernel_5=False)

        # input_to_pooling = cur_inception_in
        input_to_dense = Flatten(
                            data_format='channels_last')(inception_layer5_out)

        model_output = Dense(units=num_redshift_classes, activation='softmax')(((
                 Dense(units=1024, activation='relu')(
                       input_to_dense))))

        super().__init__(inputs=[image_input], outputs=model_output)
        opt = Adam(lr=0.001)
        self.compile(optimizer=opt,
                     loss='sparse_categorical_crossentropy',
                     metrics=['sparse_categorical_accuracy'])

    def add_inception_layer(self,
                            input_weights,
                            num_f1,
                            num_f2,
                            kernel_5=True):
        """These convolutional layers take care of the inception layer"""
        # Conv Layer 1 and Layer 2: Feed them to convolution layers 5 and 6
        c1 = Conv2D(num_f1, kernel_size=(1, 1), padding='same', activation='relu')
        c1_out = c1(input_weights)
        if kernel_5:
            c2 = Conv2D(num_f1, kernel_size=(1, 1), padding='same', activation='relu')
            c2_out = c2(input_weights)

        # Conv Layer 3 : Feed to pooling layer 1
        c3 = Conv2D(num_f1, kernel_size=(1, 1), padding='same', activation='relu')
        c3_out = c3(input_weights)

        # Conv Layer 4: Feed directly to concat
        c4 = Conv2D(num_f2, kernel_size=(1, 1), padding='same', activation='relu')
        c4_out = c4(input_weights)

        # Conv Layer 5: Feed from c1, feed to concat
        c5 = Conv2D(num_f2, kernel_size=(3, 3), padding='same', activation='relu')
        c5_out = c5(c1_out)

        # Conv Layer 6: Feed from c2, feed to concat
        if kernel_5:
            c6 = Conv2D(num_f2, kernel_size=(5, 5), padding='same', activation='relu')
            c6_out = c6(c2_out)

        # Pooling Layer 1: Feed from conv3, feed to concat
        p1 = AveragePooling2D(pool_size=(2, 2), strides=1, padding='same')
        p1_out = p1(c3_out)

        if kernel_5:
            return Concatenate()([c4_out, c5_out, c6_out, p1_out])
        else:
            return Concatenate()([c4_out, c5_out, p1_out])

    def __str__(self):
        self.summary()
        return ""

    def __repr__(self):
        return f"RedshiftInception({self.input_shape[1:]},{self.output_shape[1]})"


class RedshiftClassifierInceptionV4(Model):
    """An inception model version 4.

    This model is uses a combination of prescriptions from the
    redshift paper (http://arxiv.org/abs/1806.06607) as well as
    IncpetionV4 paper (https://arxiv.org/abs/1602.072).At its current
    form the Inception network uses hyper-parameters used by classification
    (input shape (299, 299, 3)), with PReLU activation for the convolution
    layers and all the MaxPooling layers replaced by AveragePooling.

    ToDo: Determining correct set of hyper-parameters for RedShift Classification

    Parameters
    ----------
    input_image_shape : tuple
        The shape of the input Image
    keep_probability : float, default=0.8
        The keep probability for the dropout layer
    num_redshift_bins : int
        The number of redshift classes for the end softmax layer

    Notes
    -----
    This Network assumes that the data format is `channels_last`
    """
    def __init__(self,
                 input_image_shape=(299, 299, 3),
                 keep_probability=0.8,
                 num_redshift_bins=1000):
        self.num_classes = num_redshift_bins
        image_input = Input(shape=input_image_shape)
        block_output = self.add_stem(image_input)

        # 4 Inception-A Blocks
        for block_a_index in range(1, 5):
            block_output = self.add_inception_a(block_input=block_output,
                                                block_index=block_a_index)

        # Reduction-A Block
        block_output = self.add_reduction_a(block_output)

        # 7 Inception-B Blocks
        for block_b_index in range(1, 8):
            block_output = self.add_inception_b(block_input=block_output,
                                                block_index=block_b_index)

        # Reduction-B Block
        block_output = self.add_pure_inception_reduction_b(block_output)

        # 3 Inception-C Blocks
        for block_c_index in range(1, 4):
            block_output = self.add_inception_c(block_output, block_c_index)

        # Average Pooling Block
        block_output = AveragePooling2D(pool_size=(8, 8),
                                        strides=1,
                                        padding='valid',
                                        name='final_average_pooling_block')(block_output)

        block_output = Dropout(1-keep_probability)(block_output)

        block_output = Flatten()(block_output)

        model_output = Dense(units=num_redshift_bins, activation='softmax')(block_output)

        super(RedshiftClassifierInceptionV4, self).__init__(inputs=image_input, outputs=model_output)
        opt = Adam(lr=0.001)
        self.compile(optimizer=opt,
                     loss='sparse_categorical_crossentropy',
                     metrics=['sparse_categorical_accuracy'])

    def add_stem(self, image_input):
        """Adds the stem block from Inception version 4

        The stem block is organized as shown below:

        image_input->1->2->3
        3->4
        3->5
        (4, 5)->6
        6->7->8,
        6->9->10->11->12
        (8, 12)->13->14
        13->15
        (14,15)->16

        The layers 1 through 16 are as follows:

        1. A 2DConvolution block with 32 filters, kernel size of 3 and stride of 2px, valid padding
        (`stem_conv2d_1`)
        2. Same as 1 but stride of 1px
        (`stem_conv2d_2`)
        3. A 2DConvolution block with 64 filters, kernel size of 3 and stride of 1px, same padding
        (`stem_conv2d_3`)

        4. An AveragePooling layer of pool size 3, stride of 2px
        (`stem_average_pooling_1`)

        5. A 2DConvolution block with 96 filters, kernel size of 3 and stride of 2px, valid padding
        (`stem_conv2d_4`)

        6. A Merge Concatenate layer [avg_pool_1_out, stem_conv2d_4_out]

        7. A 2DConvolution layer with 64 filters, kernel size of 1 and stride of 1px, same padding
        (`stem_conv2d_5`)
        8. A 2DConvolution layer with 96 filters, kernel size of 3 and stride of 1px, valid padding
        (`stem_conv2d_6`)

        9. A 2DConvolution layer with 64 filters, kernel size of 1 and stride of 1px, same padding
        (`stem_conv2d_7`)
        10. A 2DConvolution layer with 64 filters, kernel size of (7, 1) and stride of 1px, same padding
        (`stem_conv2d_8`)
        11. A 2DConvolution layer with 64 filters, kernel size of (1, 7) and stride of 1px, same padding
        (`stem_conv2d_9`)
        12. A 2DConvolution layer with 96 filters, kernel size of (3, 3) and stride of 1px, valid padding
        (`stem_conv2d_10`)

        13. Merge Concatenate layer [stem_conv2d_6_out, stem_conv2d_10_out]

        14. A 2D Convolution layer with 192 filters, kernel size of 3 and stride of 1px, valid padding
        (`stem_conv2d_11`)
        15. An AveragePooling layer of pool size 3, stride of 2px, valid padding
        (`stem_average_pooling_2`)

        16. A Merge Concatenate layer [stem_conv2d_11, avg_pool_2_out]

        Parameters
        ----------
        image_input : tf.Tensor
            A tensor of the initial input to the InceptionV4 Network
        """
        stem_conv2d_1_out = self.conv2d_with_prelu(image_input,
                                                   num_filters=32,
                                                   kernel_size=(3, 3),
                                                   strides=2,
                                                   padding='valid',
                                                   name='stem_conv2d_1')

        stem_conv2d_2_out = self.conv2d_with_prelu(stem_conv2d_1_out,
                                                   num_filters=32,
                                                   kernel_size=(3, 3),
                                                   strides=1,
                                                   padding='valid',
                                                   name='stem_conv2d_2')

        stem_conv2d_3_out = self.conv2d_with_prelu(stem_conv2d_2_out,
                                                   num_filters=64,
                                                   kernel_size=(3, 3),
                                                   strides=1,
                                                   padding='same',
                                                   name='stem_conv2d_3')

        avg_pool_1 = AveragePooling2D((3, 3),
                                      strides=2,
                                      padding='valid',
                                      name='stem_average_pooling_1')
        avg_pool_1_out = avg_pool_1(stem_conv2d_3_out)

        stem_conv2d_4_out = self.conv2d_with_prelu(stem_conv2d_3_out,
                                                   96,
                                                   kernel_size=(3, 3),
                                                   strides=2,
                                                   padding='valid',
                                                   name='stem_conv2d_4')

        # Concatenate
        concat_1 = Concatenate(name='stem_concatenate_1', axis=-1)
        concat_1_out = concat_1([avg_pool_1_out, stem_conv2d_4_out])

        stem_conv2d_5_out = self.conv2d_with_prelu(concat_1_out,
                                                   num_filters=64,
                                                   kernel_size=(1, 1),
                                                   padding='same',
                                                   strides=1,
                                                   name='stem_conv2d_5')

        stem_conv2d_6_out = self.conv2d_with_prelu(stem_conv2d_5_out,
                                                   num_filters=96,
                                                   kernel_size=(3, 3),
                                                   padding='valid',
                                                   strides=1,
                                                   name='stem_conv2d_6')

        stem_conv2d_7_out = self.conv2d_with_prelu(concat_1_out,
                                                   64,
                                                   kernel_size=(1, 1),
                                                   strides=1,
                                                   padding='same',
                                                   name='stem_conv2d_7')

        stem_conv2d_8_out = self.conv2d_with_prelu(stem_conv2d_7_out,
                                                   num_filters=64,
                                                   kernel_size=(7, 1),
                                                   strides=1,
                                                   padding='same',
                                                   name='stem_conv2d_8')

        stem_conv2d_9_out = self.conv2d_with_prelu(stem_conv2d_8_out,
                                                   num_filters=64,
                                                   kernel_size=(1, 7),
                                                   strides=1,
                                                   padding='same',
                                                   name='stem_conv2d_9')

        stem_conv2d_10_out = self.conv2d_with_prelu(stem_conv2d_9_out,
                                                    num_filters=96,
                                                    strides=1,
                                                    kernel_size=(3, 3),
                                                    padding='valid',
                                                    name='stem_conv2d_10')

        concat_2 = Concatenate(name='stem_concatenate_2', axis=-1)
        concat_2_out = concat_2([stem_conv2d_6_out, stem_conv2d_10_out])

        stem_conv2d_11_out = self.conv2d_with_prelu(concat_2_out,
                                                    num_filters=192,
                                                    kernel_size=(3, 3),
                                                    strides=2,
                                                    padding='valid',
                                                    name='stem_conv2d_11')
        avg_pool_2 = AveragePooling2D(pool_size=(3, 3),
                                      strides=2,
                                      padding='valid',
                                      name='stem_average_pooling_2')
        avg_pool_2_out = avg_pool_2(concat_2_out)

        return Concatenate(axis=-1)([stem_conv2d_11_out, avg_pool_2_out])

    def add_inception_a(self, block_input, block_index):
        """Adds the inception-A block from Inception version 4

        The inception-A block is organized as shown below:

        block_input->1->2 (branch 1)
        block_input->3    (branch 2)
        block_input->4->5 (branch 3)
        block_input->6->7->8 (branch 4)
        (2,3,5,8) -> 9 (layer output)

        The layers 1 through 9 are as follows:

        1. An AveragePooling layer of pool size 3, stride of 1px (`inception_a{block_index}_average_pool_1`)
        2. A 2DConvolution block with 96 filters, kernel size of 1 and stride of 1px
        same padding (`inception_a{block_index}_conv2d_1`)

        3. A 2DConvolution block with 96 filters, kernel size of 1 and stride of
        1px, same padding (`inception_a{block_index}_conv2d_2`)

        4. A 2DConvolution block with 64 filters, kernel size of 1 and stride of 1px,
        same padding (`inception_a{block_index}_conv2d_3`)
        5. A 2DConvolution block with 96 filters, kernel size of 3 and stride of 1px,
        same padding (`inception_a{block_index}_conv2d_4`)

        6. A 2DConvolution block with 64 filters, kernel size of 1 and stride of 1px,
        same padding (`inception_a{block_index}_conv2d_5`)
        7. A 2DConvolution block with 96 filters, kernel size of 3 and stride of 1px,
        same padding (`inception_a{block_index}_conv2d_6`)
        8. A 2DConvolution block with 96 filters, kernel size of 3 and stride of 1px,
        same padding (`inception_a{block_index}_conv2d_7`)

        9. A Merge Concatenate layer [a_block_branch1, a_block_branch2, a_block_branch3, a_block_branch4]

        Parameters
        ----------
        block_input : tf.Tensor
            The input to the block A
        block_index : int
            The index of this block for providing names to the layers

        Returns
        -------
        tf.Tensor
            The output of the block
        """
        a_block_branch1 = AveragePooling2D(pool_size=(3, 3),
                                           name=f'inception_a{block_index}_average_pool_1',
                                           padding='same',
                                           strides=1)(block_input)
        a_block_branch1 = self.conv2d_with_prelu(a_block_branch1,
                                                 num_filters=96,
                                                 padding='same',
                                                 kernel_size=(1, 1),
                                                 strides=1,
                                                 name=f'inception_a{block_index}_conv2d_1')

        a_block_branch2 = self.conv2d_with_prelu(block_input,
                                                 num_filters=96,
                                                 kernel_size=(1, 1),
                                                 padding='same',
                                                 strides=1,
                                                 name=f'inception_a{block_index}_conv2d_2')

        a_block_branch3 = self.conv2d_with_prelu(block_input,
                                                 num_filters=64,
                                                 kernel_size=(1, 1),
                                                 padding='same',
                                                 strides=1,
                                                 name=f'inception_a{block_index}_conv2d_3')
        a_block_branch3 = self.conv2d_with_prelu(a_block_branch3,
                                                 num_filters=96,
                                                 kernel_size=(3, 3),
                                                 padding='same',
                                                 strides=1,
                                                 name=f'inception_a{block_index}_conv2d_4')

        a_block_branch4 = self.conv2d_with_prelu(block_input,
                                                 num_filters=64,
                                                 kernel_size=(1, 1),
                                                 padding='same',
                                                 strides=1,
                                                 name=f'inception_a{block_index}_conv2d_5')
        a_block_branch4 = self.conv2d_with_prelu(a_block_branch4,
                                                 num_filters=96,
                                                 kernel_size=(3, 3),
                                                 padding='same',
                                                 strides=1,
                                                 name=f'inception_a{block_index}_conv2d_6')
        a_block_branch4 = self.conv2d_with_prelu(a_block_branch4,
                                                 num_filters=96,
                                                 kernel_size=(3, 3),
                                                 padding='same',
                                                 strides=1,
                                                 name=f'inception_a{block_index}_conv2d_7')

        return Concatenate(axis=-1)([a_block_branch1, a_block_branch2, a_block_branch3, a_block_branch4])

    def add_inception_b(self, block_input, block_index):
        """Adds the inception-B block from Inception version 4

        The inception-B block is organized as shown below:

        block_input->1->2    (branch 1)
        block_input->3       (branch 2)
        block_input->4->5->6 (branch 3)
        block_input->7->8->9->10->11 (branch 4)
        (2,3,6,11)->12  (layer output)

        The layers 1 through 12 are as follows:

        1. An AveragePooling layer of pool size 3, stride of 1px (`inception_b{block_index}_average_pool_1`)
        2. A 2DConvolution block with 128 filters, kernel size of 1 and stride of 1px
        same padding (`inception_b{block_index}_conv2d_1`)

        3. A 2DConvolution block with 384 filters, kernel size of 1 and stride of
        1px, same padding (`inception_b{block_index}_conv2d_2`)

        4. A 2DConvolution block with 192 filters, kernel size of 1 and stride of 1px,
        same padding (`inception_b{block_index}_conv2d_3`)
        5. A 2DConvolution block with 224 filters, kernel size of (7, 1) and stride of 1px,
        same padding (`inception_b{block_index}_conv2d_4`)
        6. A 2DConvolution block with 256 filters, kernel size of (1, 7) and stride of 1px,
        same padding (`inception_b{block_index}_conv2d_5`)

        7. A 2DConvolution block with 192 filters, kernel size of 1 and stride of 1px,
        same padding (`inception_b{block_index}_conv2d_6`)
        8. A 2DConvolution block with 192 filters, kernel size of (1, 7) and stride of 1px,
        same padding (`inception_b{block_index}_conv2d_7`)
        9. A 2DConvolution block with 224 filters, kernel size of (7, 1) and stride of 1px,
        same padding (`inception_b{block_index}_conv2d_8`)
        10. A 2DConvolution block with 224 filters, kernel size of (1, 7) and stride of 1px,
        same padding (`inception_b{block_index}_conv2d_9`)
        11. A 2DConvolution block with 256 filters, kernel size of (7, 1) and stride of 1px,
        same padding (`inception_b{block_index}_conv2d_10`)

        12. A Merge Concatenate layer [b_block_branch_1, b_block_branch_2, b_block_branch_3, b_block_branch_4]

        Parameters
        ----------
        block_input : tf.Tensor
            The input to the block B
        block_index : int
            The index of this block for providing names to the layers

        Returns
        -------
        tf.Tensor
            The output of the block
        """
        b_block_branch_1 = AveragePooling2D(pool_size=(3, 3),
                                            name=f'inception_b{block_index}_average_pool_1',
                                            padding='same',
                                            strides=1)(block_input)
        b_block_branch_1 = self.conv2d_with_prelu(b_block_branch_1,
                                                  num_filters=128,
                                                  kernel_size=(1, 1),
                                                  padding='same',
                                                  strides=1,
                                                  name=f'inception_b{block_index}_conv2d_1')

        b_block_branch_2 = self.conv2d_with_prelu(block_input,
                                                  num_filters=384,
                                                  kernel_size=(1, 1),
                                                  padding='same',
                                                  strides=1,
                                                  name=f'inception_b{block_index}_conv2d_2')

        b_block_branch_3 = self.conv2d_with_prelu(block_input,
                                                  num_filters=192,
                                                  kernel_size=(1, 1),
                                                  padding='same',
                                                  strides=1,
                                                  name=f'inception_b{block_index}_conv2d_3')
        b_block_branch_3 = self.conv2d_with_prelu(b_block_branch_3,
                                                  num_filters=224,
                                                  kernel_size=(7, 1),
                                                  padding='same',
                                                  strides=1,
                                                  name=f'inception_b{block_index}_conv2d_4')
        b_block_branch_3 = self.conv2d_with_prelu(b_block_branch_3,
                                                  num_filters=256,
                                                  kernel_size=(1, 7),
                                                  padding='same',
                                                  strides=1,
                                                  name=f'inception_b{block_index}_conv2d_5')

        b_block_branch_4 = self.conv2d_with_prelu(block_input,
                                                  num_filters=192,
                                                  kernel_size=(1, 1),
                                                  padding='same',
                                                  strides=1,
                                                  name=f'inception_b{block_index}_conv2d_6')
        b_block_branch_4 = self.conv2d_with_prelu(b_block_branch_4,
                                                  num_filters=192,
                                                  kernel_size=(1, 7),
                                                  padding='same',
                                                  strides=1,
                                                  name=f'inception_b{block_index}_conv2d_7')
        b_block_branch_4 = self.conv2d_with_prelu(b_block_branch_4,
                                                  num_filters=224,
                                                  kernel_size=(7, 1),
                                                  padding='same',
                                                  strides=1,
                                                  name=f'inception_b{block_index}_conv2d_8')
        b_block_branch_4 = self.conv2d_with_prelu(b_block_branch_4,
                                                  num_filters=224,
                                                  kernel_size=(1, 7),
                                                  padding='same',
                                                  strides=1,
                                                  name=f'inception_b{block_index}_conv2d_9')
        b_block_branch_4 = self.conv2d_with_prelu(b_block_branch_4,
                                                  num_filters=256,
                                                  kernel_size=(7, 1),
                                                  padding='same',
                                                  strides=1,
                                                  name=f'inception_b{block_index}_conv2d_10')

        return Concatenate(axis=-1)([b_block_branch_1, b_block_branch_2, b_block_branch_3, b_block_branch_4])

    def add_inception_c(self, block_input, block_index):
        """Adds the inception-C block from Inception version 4

        The inception-C block is organized as shown below:

        block_input->1->2 (branch 1)
        block_input->3    (branch 2)
        block_input->4->5 (branch 3)
        block_input->4->6 (branch 4)
        block_input->7->8->9->10 (branch 5)
        block_input->7->8->9->11 (branch 6)
        (2,3,5,6,10,11)->12 (layer output)

        The layers 1 through 12 are as follows:

        1. An AveragePooling layer of pool size 3, stride of 1px (`inception_c{block_index}_average_pool_1`)
        2. A 2DConvolution block with 256 filters, kernel size of 1 and stride of 1px,
        same padding (`inception_c{block_index}_conv2d_1`)

        3. A 2DConvolution block with 256 filters, kernel size of 1 and stride of 1px,
        same padding (`inception_c{block_index}_conv2d_2`)

        4. A 2DConvolution block with 384 filters, kernel size of 1 and stride of 1px,
        same padding (`incpetion_c{block_index}_conv2d_3`)

        5. A 2DConvolution block with 256 filters, kernel size of (1, 3) and stride of 1px,
        same padding (`inception_c{block_index}_conv2d_4`)

        6. A 2DConvolution block with 256 filters, kernel size of (3, 1) and stride of 1px,
        same padding (`inception_c{block_index}_conv2d_5`)

        7. A 2DConvolution block with 384 filters, kernel size of 1 and stride of 1px,
        same padding (`inception_c{block_index}_conv2d_6`)
        8. A 2DConvolution block with 448 filters, kernel size of (1, 3) and stride of 1px,
        same padding (`inception_c{block_index}_conv2d_7`)
        9. A 2DConvolution block with 512 filters, kernel size of (3, 1) and stride of 1px,
        same padding (`inception_c{block_index}_conv2d_8`)

        10. A 2DConvolution block with 256 filters, kernel size of (3, 1) and stride of 1px,
        same padding (`inception_c{block_index}_conv2d_9`)

        11. A 2DConvolution block with 256 filters, kernel size of (1, 3) and stride of 1px,
        same padding (`inception_c{block_index}_conv2d_10`)

        12. A Merge Concatenate layer [c_block_branch_1, c_block_branch_2,
        c_block_branch_3, c_block_branch_4, c_block_branch_5, c_block_branch_6]

        Parameters
        ----------
        block_input : tf.Tensor
            The input to the block C
        block_index : int
            The index of this block for providing names to the layers

        Returns
        -------
        tf.Tensor
            The output of the block
        """
        c_block_branch_1 = AveragePooling2D(pool_size=(3, 3),
                                            name=f'inception_c{block_index}_average_pool_1',
                                            strides=1,
                                            padding='same')(block_input)
        c_block_branch_1 = self.conv2d_with_prelu(c_block_branch_1,
                                                  num_filters=256,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding='same',
                                                  name=f'inception_c{block_index}_conv2d_1')

        c_block_branch_2 = self.conv2d_with_prelu(block_input,
                                                  num_filters=256,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding='same',
                                                  name=f'inception_c{block_index}_conv2d_2')

        c_block_branch_3 = c_block_branch_4 = self.conv2d_with_prelu(block_input,
                                                                     num_filters=384,
                                                                     kernel_size=(1, 1),
                                                                     strides=1,
                                                                     padding='same',
                                                                     name=f'inception_c{block_index}_conv2d_3')

        c_block_branch_3 = self.conv2d_with_prelu(c_block_branch_3,
                                                  num_filters=256,
                                                  kernel_size=(1, 3),
                                                  strides=1,
                                                  padding='same',
                                                  name=f'inception_c{block_index}_conv2d_4')

        c_block_branch_4 = self.conv2d_with_prelu(c_block_branch_4,
                                                  num_filters=256,
                                                  kernel_size=(3, 1),
                                                  strides=1,
                                                  padding='same',
                                                  name=f'inception_c{block_index}_conv2d_5')

        c_block_branch_56 = self.conv2d_with_prelu(block_input,
                                                   num_filters=384,
                                                   kernel_size=(1, 1),
                                                   strides=1,
                                                   padding='same',
                                                   name=f'inception_c{block_index}_conv2d_6')
        c_block_branch_56 = self.conv2d_with_prelu(c_block_branch_56,
                                                   num_filters=448,
                                                   kernel_size=(1, 3),
                                                   strides=1,
                                                   padding='same',
                                                   name=f'inception_c{block_index}_conv2d_7')
        c_block_branch_56 = self.conv2d_with_prelu(c_block_branch_56,
                                                   num_filters=512,
                                                   kernel_size=(3, 1),
                                                   strides=1,
                                                   padding='same',
                                                   name=f'inception_c{block_index}_conv2d_8')

        c_block_branch_5 = self.conv2d_with_prelu(c_block_branch_56,
                                                  num_filters=256,
                                                  kernel_size=(3, 1),
                                                  strides=1,
                                                  padding='same',
                                                  name=f'inception_c{block_index}_conv2d_9')

        c_block_branch_6 = self.conv2d_with_prelu(c_block_branch_56,
                                                  num_filters=256,
                                                  kernel_size=(1, 3),
                                                  strides=1,
                                                  padding='same',
                                                  name=f'inception_c{block_index}_conv2d_11')

        return Concatenate(axis=-1)([c_block_branch_1,
                                     c_block_branch_2,
                                     c_block_branch_3,
                                     c_block_branch_4,
                                     c_block_branch_5,
                                     c_block_branch_6])

    def add_reduction_a(self,
                        block_input,
                        param_k=192,
                        param_l=224,
                        param_m=256,
                        param_n=384):
        """Adds Reduction-A block from Inception version 4

        This block adds a reduction block to the Inception Network. To this end, it will convert a block grid
        of (35 * 35) into a block grid of (17 * 17). Different variants of this block (with various number of filters)
        are used with new Inception(-v4, -ResNet-v1, -ResNet-v2) variants presented in the InceptionV4 paper.
        The parameters k, l, m, n numbers represent filter bank sizes which can be looked up as:

        ..  csv-table:: Parameters k, l, m and n
            :header: "Network Variant", "k", "l", "m", "n"
            :widths: 15, 15, 15

            "Inception-V4", 192, 224, 256, 384
            "Inception-ResNet-v1", 192, 192, 256, 384
            "Inception-ResNet-v2", 256, 256, 384, 384

        The Reduction Block A is organized as shown below:

        block_input->1   (branch 1)
        block_input->2   (branch 2)
        block_input->3->4->5 (branch 3)
        (1,2,5)->6 (layer output)

        Layers 1 through 6 are as follows:

        1. An AveragePooling Layer of pool size 3, strides of 2px, valid padding (`reduction_block_a_average_pool_1`)

        2. A 2DConvolution block with `n` filters, kernel size of (3, 3), and stride of 2 px, valid padding
        (`reduction_block_a_con2d_1`)

        3. A 2DConvolution block with `k` filters, kernel size of (1, 1), and stride of 1 px, same padding
        (`reduction_block_a_con2d_2`)
        4. A 2DConvolution block with `l` filters, kernel size of (1, 1), and stride of 1 px, same padding
        (`reduction_block_a_con2d_3`)
        5. A 2DConvolution block with `m` filters, kernel size of (3, 3), and stride of 1 px, valid padding
        (`reduction_block_a_con2d_4`)

        6. A merge Concatenate layer [ reduction_a_branch_1, reduction_a_branch_2, reduction_a_branch_3 ]

        Parameters
        ----------
        block_input : tf.Tensor
            The input to this reduction block. It should of shape (35, 35, ?)
        param_k : int, default=192
            The number of filters for first convolution layer in branch3
        param_l : int, default=224
            The number of filters for second convolution layer in branch3
        param_m : int, default=256
            The number of filters for third convolution layer in branch3
        param_n : int, default=384
            The number of filters for first convolution layer in branch2
        """
        reduction_a_branch_1 = AveragePooling2D(pool_size=(3, 3),
                                                strides=2,
                                                padding='valid',
                                                name='reduction_block_a_average_pool_1')(block_input)
        reduction_a_branch_2 = self.conv2d_with_prelu(block_input,
                                                      num_filters=param_n,
                                                      kernel_size=(3, 3),
                                                      padding='valid',
                                                      strides=2,
                                                      name='reduction_block_a_conv2d_1')

        reduction_a_branch_3 = self.conv2d_with_prelu(block_input,
                                                      num_filters=param_k,
                                                      kernel_size=(1, 1),
                                                      padding='same',
                                                      strides=1,
                                                      name='reduction_block_a_conv2d_2')
        reduction_a_branch_3 = self.conv2d_with_prelu(reduction_a_branch_3,
                                                      num_filters=param_l,
                                                      kernel_size=(3, 3),
                                                      padding='same',
                                                      strides=1,
                                                      name='reduction_block_a_conv2d_3')
        reduction_a_branch_3 = self.conv2d_with_prelu(reduction_a_branch_3,
                                                      num_filters=param_m,
                                                      kernel_size=(3, 3),
                                                      padding='valid',
                                                      strides=2,
                                                      name='reduction_block_a_con2d_4')
        return Concatenate()([reduction_a_branch_1, reduction_a_branch_2, reduction_a_branch_3])

    def add_pure_inception_reduction_b(self, block_input):
        """Add reduction-B block from Inception version 4 (used by pure Inception V4)

        The reduction block B has a schema for reducing 17 * 17 grid to 8 * 8
        grid. This is used by pure InceptionV4 Network.

        The Reduction block-B is organized as shown below:

        block_input->1     (branch 1)
        block_input->2->3  (branch 2)
        block_input->4->5->6->7  (branch 3)
        (1,3,7)->9  (layer output)

        Layers 1 through 7 are as follows:

        1. An AveragePooling Layer of pool size 3, strides of 2px, valid padding (`reduction_block_b_average_pool_1`)

        2. A 2DConvolution block with 192 filters, kernel size of (1, 1), and stride of 1 px, same padding
        (`reduction_block_b_conv2d_1`)
        3. A 2DConvolution block with 192 filters, kernel size of (3, 3), and stride of 2 px, valid padding
        (`reduction_block_b_conv2d_2`)

        4. A 2DConvolution block with 256 filters, kernel size of (1, 1), and stride of 1 px, same padding
        (`reduction_block_b_conv2d_3`)
        5. A 2DConvolution block with 256 filters, kernel size of (1, 7), and stride of 1 px, same padding
        (`reduction_block_b_conv2d_4`)
        6. A 2DConvolution block with 320 filters, kernel size of (7, 1), and stride of 1 px, same padding
        (`reduction_block_b_conv2d_5`)
        7. A 2DConvolution block with 320 filters, kernel size of (3, 3), and stride of 2 px, valid padding
        (`reduction_block_b_conv2d_6`)

        8. A Merge Concatenate Layer [reduction_b_branch_1, reduction_b_branch_2, reduction_b_branch_3]

        Parameters
        ----------
        block_input : tf.Tensor
            The input to this reduction block. It should be of shape (17, 17, ?)
        """
        reduction_b_branch_1 = AveragePooling2D(pool_size=(3, 3),
                                                padding='valid',
                                                strides=2,
                                                name='reduction_block_b_average_pool_1')(block_input)

        reduction_b_branch_2 = self.conv2d_with_prelu(block_input,
                                                      num_filters=192,
                                                      kernel_size=(1, 1),
                                                      strides=1,
                                                      padding='same',
                                                      name='reduction_block_b_conv2d_1')
        reduction_b_branch_2 = self.conv2d_with_prelu(reduction_b_branch_2,
                                                      num_filters=192,
                                                      kernel_size=(3, 3),
                                                      strides=2,
                                                      padding='valid',
                                                      name='reduction_block_b_conv2d_2')

        reduction_b_branch_3 = self.conv2d_with_prelu(block_input,
                                                      num_filters=256,
                                                      kernel_size=(1, 1),
                                                      strides=1,
                                                      padding='same',
                                                      name='reduction_block_b_conv2d_3')
        reduction_b_branch_3 = self.conv2d_with_prelu(reduction_b_branch_3,
                                                      num_filters=256,
                                                      kernel_size=(1, 7),
                                                      strides=1,
                                                      padding='same',
                                                      name='reduction_block_b_conv2d_4')
        reduction_b_branch_3 = self.conv2d_with_prelu(reduction_b_branch_3,
                                                      num_filters=320,
                                                      kernel_size=(7, 1),
                                                      strides=1,
                                                      padding='same',
                                                      name='reduction_block_b_conv2d_5')
        reduction_b_branch_3 = self.conv2d_with_prelu(reduction_b_branch_3,
                                                      num_filters=320,
                                                      kernel_size=(3, 3),
                                                      strides=2,
                                                      padding='valid',
                                                      name='reduction_block_b_conv2d_6')

        return Concatenate(axis=-1)([reduction_b_branch_1, reduction_b_branch_2, reduction_b_branch_3])

    def __str__(self):
        self.summary()
        return ""

    def __repr__(self):
        return f"RedshiftClassifierInceptionV4({self.input_shape[1:]},{self.output_shape[1]})"

    @staticmethod
    def conv2d_with_prelu(layer_input,
                          num_filters,
                          kernel_size=(3, 3),
                          padding='same',
                          strides=1,
                          name=None):
        """Process a convolution layer with parametric relu Activation"""
        conv2d = Conv2D(filters=num_filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        name=name,
                        activation='linear')
        conv2d_out = PReLU()(conv2d(layer_input))
        return conv2d_out
