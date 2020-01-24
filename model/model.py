from keras import Model
from keras.layers import (Input, Conv2D, AveragePooling2D,
                          concatenate, Dense, PReLU,
                          Flatten, Dropout, Add, GlobalAveragePooling2D)
from keras.optimizers import Adam
import numpy as numpy

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

        super(RedshiftClassifierResNet, self).__init__(
                inputs=[image_input],outputs=model_output)
        self.summary()
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

def main():
    test_model = RedshiftClassifierResNet((128, 128, 5), 32)

if __name__ == "__main__":
    main()