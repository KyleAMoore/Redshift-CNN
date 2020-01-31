import numpy as np
from model.model import RedshiftClassifierResNet
from time import time

class Train():
    def __init__(self, batch_size, epochs, validation_split):
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_split = validation_split
    
    def execute(self, model, inputs, labels):
        print("Beginning model training")
        return model.fit(inputs,labels,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         validation_split=self.val_split,
                         verbose=1)

def save_model(model, history, model_label, data_label, batch_size, epochs, val_split):
    print("Saving model weights and training history")
    file_name = f'model/saved/{model_label}-{data_label}-bs{batch_size}-e{epochs}-vs{str(val_split)[2:]}'
    model.save_weights(file_name+'.h5')
    with open(file_name+'.hist','wb') as pkl:
        dump(history.history,pkl)

if __name__ == "__main__":
    from pickle import load, dump
    with open('data/cifar-10/prep/cifar-10-train.pkl','rb') as pkl:
        train_imgs = load(pkl)
        train_labels = load(pkl)
    
    with open('data/cifar-10/prep/cifar-10-test.pkl','rb') as pkl:
        test_imgs = load(pkl)
        test_labels = load(pkl)

    batch_size = 32
    epochs = 25
    val_split = 0.15

    model = RedshiftClassifierResNet((32,32,3), 10)
    trainer = Train(batch_size, epochs, val_split)
    
    start = time()
    hist = trainer.execute(model, train_imgs, train_labels)
    elapsed = time() - start
    hist.history['train_time'] = elapsed

    save_model(model, hist,
               'resnet',
               'cf10',
               batch_size,
               epochs,
               val_split)