from abc import ABC, abstractmethod
from keras import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import (Activation, Add, Average, AveragePooling2D,
                          BatchNormalization, Concatenate, Conv2D,
                          Dense, Flatten, Input)
from keras.metrics import Metric
from keras.optimizers import Adam
from keras.utils import print_summary, plot_model, Sequence
import numpy as np
from os import makedirs
from random import randint, random, shuffle
import tensorflow as tf

class RedshiftClassifier(ABC, Model):
    """Allows defining of redshift estimation neural networks. In order to extend this abstract
    class with a specific architecture, create a class that inherits this class. This subclass
    at a minimum needs to define the build function, which must return a tuple that contains two
    lists of keras layers, with the first list containing all input layers and the second list
    containing all output layers.
    
    This abstract base class was created with the assumption that the functional keras API is used
    to build architectures. Other paradigms for creating models have not been tested and may not
    perform correctly.
    """
    def __init__(self, input_shape, num_bins=180, max_val=0.4, **kwargs):

        inputs, outputs = self.build(input_shape, num_bins, **kwargs)
        super().__init__(inputs=inputs, outputs=outputs)

        self.num_bins = num_bins
        self.max_val = max_val
        
        self.compile(optimizer=Adam(lr=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['sparse_categorical_accuracy',
                              PredictionBias(rs_num_bins=self.num_bins, rs_max_val=self.max_val),
                              DeviationMAD(rs_num_bins=self.num_bins, rs_max_val=self.max_val),
                              FractionOutliers(rs_num_bins=self.num_bins, rs_max_val=self.max_val),
                              AverageCRPS(rs_num_bins=self.num_bins, rs_max_val=self.max_val)])

    @abstractmethod
    def build(self, input_shape, num_bins, **kwargs):
        pass

    def predict_redshift(self, imgs):
        """Predicts the redshift value of a set of galaxy images.

        Args:
            imgs (np.array(X, 64, 64, 5)): Array of galaxy images

        Returns:
            np.array: predicted redshift values
        """
        pdfs = self.predict(imgs)
        return self.pdf_to_redshift(pdfs)

    def pdf_to_redshift(self, pdfs):
        step = self.max_val / self.num_bins
        bin_starts = np.arange(0, self.max_val, step)
        return np.sum((bin_starts + (step / 2)) * pdfs, axis=1)

    def train(self,
              imgs,
              labels,
              batch_size = 32,
              epochs = 30,
              val_split = 0.15,
              checkpoint_dir = 'checkpoints',
              rotate_chance = 0.4,
              flip_chance = 0.2,
              eval_every_epoch = False,
              test_imgs = None,
              test_labels = None):
        """Trains the redshift model on the provided galaxy images.

        Args:
            imgs (np.array(X,64,64,5)): Array of galaxy images.
            labels (np.array(X)): Array of true redshift values.
            batch_size (int, optional): Number of images to include in every training batch. Defaults to 32.
            epochs (int, optional): Number of epochs to train the model. Defaults to 30.
            val_split (floa, optional): Percentage of the training data to be used for validation. Recommended that this be
                                        changed to 0 when eval_every_epoch is True. Defaults to 0.15.
            checkpoint_dir (str, optional): Directory where training checkpoints should be saved. Defaults to 'checkpoints'.
            rotate_chance (float, optional): Probability that a given image will be rotated at the beginning of every epoch.
                                             Rotations are equally likely to be 90, 180, or 270 degree rotations.This probability
                                             is applied independently to every image every epoch. Defaults to 0.4.
            flip_chance (float, optional): Probability that a given image will be flipped at the beginning of every epoch. Flips
                                           can happen along either axis with equiprobabilty, but will not flip along both in the
                                           same epoch. This probability is applied independently to every image every epoch.
                                           Defaults to 0.2.
            eval_every_epoch (bool, optional): Whether to evaluate the model at every epoch. If this is True, test_imgs and
                                               test_labels must not be None. Defaults to False.
            test_imgs (np.array(X,64,64,5), optional): Galaxy images for evaluation. Defaults to None.
            test_labels (np.array(X), optional): True redshift values for evaluation. Defaults to None.

        Raises:
            ValueError: Raised if test_imgs or test_labels are not defined when eval_every_epoch is True
        """

        makedirs(checkpoint_dir, exist_ok=True)

        callbacks = [
            ModelCheckpoint(checkpoint_dir + '/weights.{epoch:02d}.hdf5'),
        ]
        if eval_every_epoch:
            if test_imgs is None or test_labels is None:
                raise ValueError('If eval_every_epoch is True, a list of images and redshift values must also be provided using the keyword arguments test_imgs and test_labels')
            else:
                callbacks.append(EvalEveryEpoch())
        
        cat_labels = labels // (self.max_val / self.num_bins)

        indices = list(range(len(imgs)))
        shuffle(indices)
        val_ind  = indices[:int(len(imgs)*val_split)]
        train_ind = indices[int(len(imgs)*val_split):]

        train_imgs = np.take(imgs, train_ind, axis=0)
        train_labels = np.take(cat_labels, train_ind)

        val_imgs = np.take(imgs, val_ind, axis=0)
        val_labels = np.take(cat_labels, val_ind)
        
        train_seq = ImageSequence(train_imgs, train_labels, batch_size, rotate_chance, flip_chance)
        val_seq = ImageSequence(val_imgs, val_labels, batch_size, 0, 0)

        self.fit_generator(train_seq,
                           validation_data=val_seq,
                           epochs=epochs,
                           verbose=1,
                           callbacks=callbacks)

    def model_graph(self, file_path):
        plot_model(self, file_path, show_shapes=True)

    def __str__(self):
        lines = []
        self.summary(print_fn=lambda x: lines.append(x))
        return '\n'.join(lines)

    def __repr__(self):
        return "{name}({input},{output})".format(name=type(self).__name__,
                                                 input=self.input_shape[1:],
                                                 output=self.output_shape[1])

class ImageSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size=32, rotate_chance=0.2, flip_chance=0.2):
        """Generator for network inputs and outputs. Provides the ability to randomly mutate
           images after each epoch.

           At the end of each epoch, every image in the x_set has a chance to be mutated via
           flipping and/or rotating. Flipping is equally likely to be preformed over either
           axis (not both) and rotation is equally likely to be 90, 180, or 270 degrees
        
        Args:
            x_set (numpy.array): array of inputs to the network
            y_set (numpy.array): array of ground-truth outputs
            batch_size (int, optional): Defaults to 32.
            rotate_chance (float, optional): Probability for each image to be individually flipped after
                each epoch (randomly chooses axis to flip across). Defaults to 0.4.
            flip_chance (float, optional): Probability for each image to be individually rotated after
                each epoch (randomly chooses degree of rotation). Defaults to 0.2.
        """    
        self.x, self.y = np.array(x_set), np.array(y_set)
        self.batch_size = batch_size
        self.rot_chance = rotate_chance
        self.flip_chance = flip_chance

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def mutate(self, img):    
        if random() < self.rot_chance:
            img = np.rot90(img,randint(1,3),(0,1))
        if random() < self.flip_chance:
            img = np.flip(img,randint(0,1))
        return img

    def on_epoch_end(self):
        self.x = np.array([self.mutate(i) for i in self.x])

class RedshiftMetric(Metric):
    """Evaluates the model using the metrics defined in https://arxiv.org/abs/1806.06607
    """       
    def __init__(self, name='def_redshift_metric', rs_num_bins=32, rs_max_val=3.5, **kwargs):
        super(RedshiftMetric, self).__init__(name=name, **kwargs)
        self.rs_num_bins = rs_num_bins
        self.rs_max_val = rs_max_val

    def residuals(self, y_true, y_pred):
        step = self.rs_max_val / self.rs_num_bins
        bins = np.arange(0, self.rs_max_val, step) + (step / 2)
        tf.reduce_sum(tf.multiply(y_pred, bins), axis=1)

        return (y_pred - y_true) / (y_true + 1)

    def result(self):
        return self.value

class PredictionBias(RedshiftMetric):
    """Calculates the bias of the model. Bias here is defined as the average distance
    (and direction) of predictions from the true redshift value.

    Args:
        y_true (tf.Tensor)
        y_pred (tf.Tensor)

    Returns:
        tf.Tensor: Tensor representing the average crps value
    """
    def __init__(self, name='pred_bias', **kwargs):
        super(PredictionBias, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight = None):
        residuals = self.residuals(y_true, y_pred)
        self.value = tf.math.reduce_mean(residuals)

class DeviationMAD(RedshiftMetric):
    """Calculates the MAD deviation of the predictions.

    Args:
        y_true (tf.Tensor)
        y_pred (tf.Tensor)

    Returns:
        tf.Tensor: Tensor representing the average crps value
    """
    def __init__(self, name='MAD_dev', **kwargs):
        super(DeviationMAD, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight = None):
        residuals = self.residuals(y_true, y_pred)
        res_med = tf.numpy_function(np.median, [residuals], residuals.dtype)

        self.value = tf.numpy_function(np.median, [tf.abs(residuals - res_med)], residuals.dtype) * 1.4826

class FractionOutliers(RedshiftMetric):
    """Calculates the percentage of outputs that are considered outliers. As in the source paper,
    outliers are defined as redshift values that are more than 5 times the MAD deviation away from
    the mean.

    Args:
        y_true (tf.Tensor)
        y_pred (tf.Tensor)

    Returns:
        tf.Tensor: Tensor representing the average crps value
    """
    def __init__(self, name='frac_outliers', **kwargs):
        super(FractionOutliers, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight = None):
        residuals = self.residuals(y_true, y_pred)
        res_med = tf.numpy_function(np.median, [residuals], residuals.dtype)
        dev_MAD = tf.numpy_function(np.median, [tf.abs(residuals - res_med)], residuals.dtype) * 1.4826

        outliers = tf.greater(tf.abs(residuals), dev_MAD * 5)
        self.value = tf.reduce_mean(tf.cast(outliers, residuals.dtype))

class AverageCRPS(RedshiftMetric):
    """Calculates the Continous Ranked Probability Score of the prediciton.

    This implementation was originally developed by Stanislav Arnaudov for the
    Edward repository by Blei Lab. The source code, which as of 5/11/2020 has sat
    as an unpulled commit to the repo for 2 years (the repo appears to have
    stagnated as of 7/24/2018 and the commit was added on 10/18/2018), can be
    found at https://github.com/blei-lab/edward/pull/922/files

    Args:
        y_true (tf.Tensor)
        y_pred (tf.Tensor)

    Returns:
        tf.Tensor: Tensor representing the average crps value
    """
    def __init__(self, name='avg_crps', **kwargs):
        super(AverageCRPS, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight = None):
        score = tf.reduce_mean(tf.abs(tf.subtract(y_pred, y_true)), axis=-1)
        diff = tf.subtract(tf.expand_dims(y_pred, -1), tf.expand_dims(y_pred, -2))
        score = tf.add(score, tf.multiply(tf.constant(-0.5, dtype=diff.dtype),tf.reduce_mean(tf.abs(diff),axis=(-2, -1))))

        self.value = tf.reduce_mean(score)

class EvalEveryEpoch(Callback):
    """Allows evaluation of the model at every epoch of training. Useful for tracking the
    training of the model for insight into how quickly the model converges or if it tends
    to overfit at a certain stage of training.
    """

    def __init__(self, test_imgs, test_labels):
        super(EvalEveryEpoch, self).__init__()
        self.imgs = test_imgs
        self.labels = test_labels

    def on_train_begin(self, logs=None):
        self.train_hist = []

    def on_epoch_end(self, epoch, logs=dict()):
        results = self.model.evaluate(self.imgs, self.labels)
        self.train_hist.append(logs)
        for met_name, met_val in results.items():
            self.train_hist[-1]['test_' +  met_name] = met_val

    def on_train_end(self, logs=None):
        self.model.history = self.train_hist

class RedshiftClassifierResNet(RedshiftClassifier):
    def __init__(self,
                 input_shape,
                 num_bins=32,
                 max_val=0.4,
                 num_res_blocks=6,
                 num_res_stacks=4,
                 init_num_filters=16):
        super().__init__(input_shape,
                         num_bins,
                         max_val,
                         num_res_blocks=num_res_blocks,
                         num_res_stacks=num_res_stacks,
                         init_num_filters=init_num_filters)

    def build(self, input_shape, num_bins, **kwargs):
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
        num_res_blocks = kwargs.get('num_res_blocks', 6)
        num_res_stacks = kwargs.get('num_res_stacks', 4)
        init_num_filters = kwargs.get('init_num_filters', 16)

        # Input Layer Galactic Images
        image_input = Input(shape=input_shape)
        
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
        model_output = Dense(units=num_bins, activation='softmax')(
                Dense(units=1024, activation='relu')(input_to_dense))
    
        return image_input, model_output

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

class RedshiftClassifierInception(RedshiftClassifier):
    """
        This class is adapted from the model written by Umesh Timalsina
        of the Institute for Software Integrated Systems. The original
        code was pulled from https://github.com/umesh-timalsina/redshift/blob/master/model/model.py
        on 2/5/2020
    """
    def __init__(self, input_shape, num_bins=32, max_val=0.4):
        super(RedshiftClassifierInception, self).__init__(input_shape, num_bins, max_val)

    def build(self, input_shape, num_bins, **kwargs):
        # Input Layer Galactic Images
        image_input = Input(shape=input_shape)
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

        model_output = Dense(units=num_bins, activation='softmax')(((
                 Dense(units=1024, activation='relu')(
                       input_to_dense))))

        return image_input, model_output

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
