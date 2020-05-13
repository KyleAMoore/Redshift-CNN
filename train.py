import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import Sequence
from random import randint, random, shuffle
from pickle import load, dump
import os

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

class Train():
    def __init__(self, batch_size=32, epochs=10, val_split=0.15, num_bins=32, max_val=0.4):
        """Trainer object for training a keras model. Specialized for redshift estimation.
        
        Args:
            batch_size (int, optional): Defaults to 32.
            epochs (int, optional): Defaults to 10.
            val_split (float, optional): Percent of the data to use for validation. Defaults to 0.15.
            num_bins (int, optional): Number of bins to use in model output. Defaults to 32.
            max_val (float, optional): Maximum expected redshift value. Necessary for categorical
                conversion. Defaults to 3.5.
        """    
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_split = val_split
        self.num_bins = num_bins
        self.max_val = max_val

    def train_val_split(self, imgs, labels):
        indices = list(range(len(imgs)))
        shuffle(indices)
        val_ind  = indices[:int(len(imgs)*self.val_split)]
        train_ind = indices[int(len(imgs)*self.val_split):]

        val_set = (np.take(imgs,   val_ind, axis=0), np.take(labels,   val_ind))
        train_set = (np.take(imgs, train_ind, axis=0), np.take(labels, train_ind))

        return (train_set, val_set)

    def execute(self, model, inputs, labels, checkpoint_folder, rotate_chance=0.4, flip_chance=0.2):
        """Begins training session
        
        Args:
            model (keras.Model): The model architecture to be trained. Should be an already compiled
                keras model.
            inputs (numpy.array): Inputs to train model on
            labels (numpy.array): True desired values of training inputs
            checkpoint_folder (str): folder in which to save training weights and history. Folder must
                exist prior to execution
            rotate_chance (float, optional): Probability for each image to be individually flipped after
                each epoch (randomly chooses axis to flip across). Defaults to 0.4.
            flip_chance (float, optional): Probability for each image to be individually rotated after
                each epoch (randomly chooses degree of rotation). Defaults to 0.2.
        
        Returns:
            keras.callbacks.History: history object containing the training and validation metrics at
                the end of each epoch.
        """        
        cat_labels = self.to_categorical(labels)
        
        (train_imgs, train_labels), (val_imgs, val_labels) = self.train_val_split(inputs, cat_labels)

        train_seq = ImageSequence(train_imgs, train_labels, self.batch_size, rotate_chance, flip_chance)
        val_seq = ImageSequence(val_imgs, val_labels, self.batch_size, 0, 0)

        callbacks = [
            ModelCheckpoint(checkpoint_folder + '/weights.{epoch:02d}.hdf5'),
        ]

        return model.fit_generator(train_seq,
                                   validation_data=val_seq,
                                   epochs=self.epochs,
                                   verbose=1,
                                   callbacks=callbacks)
    
    def to_categorical(self, labels):
        return labels // (self.max_val / self.num_bins)

def train_model(model,
                train_imgs,
                train_labels,
                arch_label,
                data_label,
                batch_size=32,
                epochs=10,
                num_rs_bins=32,
                max_rs_val=0.4,
                val_split=0.15,
                rot_chance=0.4,
                flip_chance=0.2):
    """Runs the training pipeline for the specified model.
    
    Args:
        model (keras.Model): The model architecture to be trained. Should be an already compiled
            keras model
        train_imgs (numpy.array): Images to train model on
        train_labels (numpy.array): Redshift values of training images
        arch_label (str): Name of architecture being tested. A folder with this name needs to
            exist in the location %Run_dir%/saved/
        data_label (str): Name of the dataset to be trained on
        batch_size (int, optional): Defaults to 32.
        epochs (int, optional): Defaults to 10.
        num_rs_bins (int, optional): Number of bins to use in model output (generates a pdf of this
            length). Defaults to 32.
        max_rs_val (float, optional): Maximum expected redshift value. Necessary for categorical
            conversion. Defaults to 0.4.
        val_split (float, optional): Percent of the data to use for validation. Defaults to 0.15.
        rot_chance (float, optional): Probability for each image to be individually flipped after
            each epoch (randomly chooses axis to flip across). Defaults to 0.4.
        flip_chance (float, optional): Probability for each image to be individually rotated after
            each epoch (randomly chooses degree of rotation). Defaults to 0.2.
    """    

    trainer = Train(batch_size, epochs, val_split, num_rs_bins, max_rs_val)

    directory = 'model/saved/'+arch_label
    if not os.path.exists(directory):
        os.makedirs(directory)

    print("Beginning training of model: " + arch_label)
    print("Model history and weights will be saved in location: " + directory)
    hist = trainer.execute(model,
                           train_imgs,
                           train_labels, 
                           directory,
                           rotate_chance=rot_chance,
                           flip_chance=flip_chance)

    file_name = '{directory}/{data_label}.hist'.format(directory=directory, data_label=data_label)
    with open(file_name,'wb') as pkl:
        dump(hist.history, pkl)

def main(mode=0):
    """
        Runs one of a selection of preset evaluation sets.

        Argument mode must be an integer in the range [0,1]

        1 - trains a resnet model with an inception model with both using their default
            hyperparameters. (DEFAULT)

        2 - trains of resnet models of different topology. Trains a total of 15 models which
            combined have every combination of values of num_res_blocks and num_res_stacks in the
            ranges [4,8] and [3,5] respectively
    """
    from model.model import RedshiftClassifierResNet, RedshiftClassifierInception

    with open('data/SDSS/prep/sdss-train.pkl','rb') as pkl:
        train_imgs = load(pkl)
        train_labels = load(pkl)

    image_shape = (64,64,5)
    num_classes = 32

    if mode == 0:
        model = RedshiftClassifierResNet(image_shape, num_classes, num_res_blocks=6, num_res_stacks=4)
        train_model(model, train_imgs, train_labels, 'resnet', 'SDSS', epochs=15)
        model = RedshiftClassifierInception(image_shape, num_classes)
        train_model(model, train_imgs, train_labels, 'incep', 'SDSS', epochs=15)

    elif mode == 1:
        for num_blocks in range(4,9):
            for num_stacks in range(3,6):
                model = RedshiftClassifierResNet(image_shape, num_classes,
                                                 num_res_blocks=num_blocks,
                                                 num_res_stacks=num_stacks)
                model_label = 'resnet-{b}-{s}'.format(b=num_blocks, s=num_stacks)
                train_model(model, train_imgs, train_labels, model_label, 'SDSS', epochs=10)

    else:
        raise ValueError('Mode must be an integer in the range [0,1] (default=0)')
