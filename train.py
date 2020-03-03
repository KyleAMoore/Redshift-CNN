"""
    Usage:
        python train.py [args]

    Arguments:
        -r    -    Train ResNet model
        -i    -    Train Inception model

"""

import numpy as np
from model.model import RedshiftClassifierResNet, RedshiftClassifierInception
from keras.callbacks import ModelCheckpoint
from time import time
from sys import argv

class Train():
    def __init__(self, batch_size, epochs, validation_split):
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_split = validation_split
    
    def execute(self, model, inputs, labels, checkpoint_folder):
        return model.fit(inputs,labels,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         validation_split=self.val_split,
                         verbose=1,
                         callbacks=[ModelCheckpoint(checkpoint_folder + '/weights.{epoch:02d}_{val_sparse_categorical_accuracy:.2f}.hdf5')])

def save_model(model, history, model_label, data_label, batch_size, epochs, val_split):
    print("Saving model weights and training history")
    file_name = f'model/saved/{model_label}/{data_label}-bs{batch_size}-e{epochs}-vs{str(val_split)[2:]}'
    model.save_weights(file_name+'.h5')
    with open(file_name+'.hist','wb') as pkl:
        dump(history.history,pkl)

def main(model, arch_label, train_imgs, train_labels, data_label, batch_size=32, epochs=15, val_split=0.15):
    trainer = Train(batch_size, epochs, val_split)
    
    start = time()
    print("Beginning training of model: " + arch_label)
    hist = trainer.execute(model, train_imgs, train_labels, 'model/saved/'+arch_label)
    hist.history['train_time'] = time() - start

    save_model(model, hist,
               arch_label,
               data_label,
               batch_size,
               epochs,
               val_split)

if __name__ == "__main__":
    from pickle import load, dump
    with open('data/cifar-10/prep/cifar-10-train.pkl','rb') as pkl:
        train_imgs = load(pkl)
        train_labels = load(pkl)
        input_shape = (32,32,3)
        num_classes = 10

    archs : list = []
    labels : list = []
    for i in range(1,len(argv)):
        arg = argv[i].lower()
        if arg == '-i':
            archs.append(RedshiftClassifierInception(input_shape, num_classes))
            labels.append("incep")
        elif arg == '-r':
            archs.append(RedshiftClassifierResNet(input_shape, num_classes))
            labels.append("resnet")
        else:
            print(f"Illegal Argument: {arg}")
            exit(1)

    for i in range(len(archs)):
        main(archs[i], labels[i], train_imgs, train_labels, 'cifar10')