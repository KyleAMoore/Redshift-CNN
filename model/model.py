from keras import Model
from keras.layers import (Input, Conv2D, AveragePooling2D,
                          concatenate, Dense, PReLU,
                          Flatten, Dropout, Add, GlobalAveragePooling2D)
from keras.optimizers import Adam
from keras.utils import print_summary

class RedshiftClassifierResNet(Model):
    def __init__(self,
                 input_img_shape,
                 num_redshift_classes):
        """Initializes the ResNet model"""

        # Input Layer Galactic Images
        image_input = Input(shape=input_img_shape)

        # Initial Convolution Layer
        conv_1 = Conv2D(64,
                        kernel_size=(7,7),
                        # strides=2,
                        padding='same',
                        activation='relu')
        conv_1_out = conv_1(image_input)

        # Initial Pooling Layer
        pooling_layer1 = AveragePooling2D(pool_size=(2, 2),
                                        #   strides=2,
                                          padding='same')
        pooling_layer1_out = pooling_layer1(conv_1_out)

        # # Residual Blocks
        residual_block1_out = self.add_residual_block(pooling_layer1_out,
                                                      filter_size=64,
                                                      num_layers=3)

        residual_block2_out = self.add_residual_block(residual_block1_out,
                                                      filter_size=128,
                                                      num_layers=4)

        residual_block3_out = self.add_residual_block(residual_block2_out,
                                                      filter_size=256,
                                                      num_layers=6)

        residual_block4_out = self.add_residual_block(residual_block3_out,
                                                      filter_size=512,
                                                      num_layers=3)

        # End Pooling Layer
        pooling_layer2 = AveragePooling2D(pool_size=(2, 2),
                                          padding='same')
        pooling_layer2_out = pooling_layer2(residual_block4_out)

        # Original paper uses GlobalAveragePooling in lieu of avg pool + dense,
        # may be worth implementing and comparing
        # pooling_layer2_out = GlobalAveragePooling2D()(residual_block4_out)

        # Fully Connected Layers
        input_to_dense = Flatten(data_format='channels_last')(pooling_layer2_out)
        model_output = Dense(units=num_redshift_classes, activation='softmax')(((
                Dense(units=1024, activation='relu')(
                       input_to_dense))))

        super().__init__(inputs=[image_input],outputs=model_output)
        opt = Adam(lr=0.001)
        self.compile(optimizer=opt,
                     loss='sparse_categorical_crossentropy',
                     metrics=['sparse_categorical_accuracy'])

    def add_residual_layer(self,
                           input_weights,
                           filter_size,
                           init_stride_2=False):
        """Adds a single residual group (composed of 2 layers each)"""
        
        c1 = Conv2D(filter_size,
                    kernel_size=(3,3),
                    strides=(2 if init_stride_2 else 1),
                    padding='same',
                    activation='relu')
        c1_out = c1(input_weights)

        c2 = Conv2D(filter_size, kernel_size=(3,3), padding='same',activation='relu')
        c2_out = c2(c1_out)

        shortcut = Conv2D(filter_size,
                          kernel_size=(1,1),
                          strides=(2 if init_stride_2 else 1),
                          padding='same',
                          activation='relu')
        shortcut_out = shortcut(input_weights)
        
        return Add()([c2_out, shortcut_out])

    def add_residual_block(self,
                           input_weights,
                           filter_size,
                           num_layers,
                           zero_padded=True):
        """Adds a block composed of multiple residual layers"""
        
        output_weights = input_weights
        output_weights = self.add_residual_layer(output_weights,
                                                 filter_size=filter_size,
                                                 init_stride_2=True)
        for _ in range(num_layers-1):
            output_weights = self.add_residual_layer(output_weights,
                                                     filter_size=filter_size,
                                                     init_stride_2=False)

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
        #TODO: Make match original inception network for comparison ot ResNet
        #TODO: comparre with keras-application inception-v3

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
            return concatenate([c4_out, c5_out, c6_out, p1_out])
        else:
            return concatenate([c4_out, c5_out, p1_out])

    def __str__(self):
        self.summary()
        return ""

    def __repr__(self):
        return f"RedshiftInception({self.input_shape[1:]},{self.output_shape[1]})"


def main():
    test_model_res = RedshiftClassifierResNet((128, 128, 5), 32)
    test_model_inc = RedshiftClassifierInception((128, 128, 5), 32)

    print(test_model_res.__repr__())
    print(test_model_res)

    print(test_model_inc.__repr__())
    print(test_model_inc)

if __name__ == "__main__":
    main()