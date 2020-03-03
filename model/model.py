from keras import Model
from keras.layers import (Input, Conv2D, AveragePooling2D,
                          Flatten, Dense, Activation, Add,
                          Concatenate, BatchNormalization)
from keras.optimizers import Adam
from keras.utils import print_summary

class RedshiftClassifierResNet(Model):
    def __init__(self,
                 input_img_shape,
                 num_redshift_classes,
                 num_res_blocks=8,
                 num_res_stacks=3,
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

        # Original paper uses GlobalAveragePooling in lieu of avg pool + dense,
        # may be worth implementing and comparing
        # pooling_layer2_out = GlobalAveragePooling2D()(residual_block4_out)

        # End Pooling Layer
        pooling_layer = AveragePooling2D(pool_size=int(weights.shape[1]))
        pooling_layer_out = pooling_layer(weights)

        # Fully Connected Layers
        input_to_dense = Flatten(data_format='channels_last')(pooling_layer_out)
        model_output = Dense(units=num_redshift_classes, activation='softmax')(((
                Dense(units=1024, activation='relu')(
                       input_to_dense))))

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

        #TODO: Change to use max pooling
        #TODO: Make match original inception network for comparison to ResNet
        #TODO: compare with keras-application inception-v3

        # input_to_pooling = cur_inception_in
        input_to_dense = Flatten(
                            data_format='channels_last')(inception_layer5_out)
        print(input_to_dense.shape)
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

if __name__ == "__main__":
    test_model_res = RedshiftClassifierResNet((128, 128, 5), 32)
    test_model_inc = RedshiftClassifierInception((128, 128, 5), 32)

    print(test_model_res.__repr__())
    print(test_model_res)

    print(test_model_inc.__repr__())
    print(test_model_inc)
