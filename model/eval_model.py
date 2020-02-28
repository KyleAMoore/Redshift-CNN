import numpy as np
from matplotlib import pyplot as plt
from model import RedshiftClassifierResNet, RedshiftClassifierInception
from pickle import dump, load
from glob import glob, iglob
from time import time


def test_model(directory, model, test_imgs, test_labels):
    with open(glob(directory+'*.hist')[0],'rb') as pkl:
        hist = load(pkl)

    incepModel = model((32,32,3), 10)
    i = 0
    for weight_file in iglob(directory + 'weights.*.hdf5'):
        i += 1
        print('Testing model saved in location: ' + weight_file)
        incepModel.load_weights(weight_file)
        start = time()
        res = incepModel.evaluate(test_imgs, test_labels)
        test_time = time() - start
        try:
            hist['test_loss'].append(res[0])
            hist['test_sparse_categorical_accuracy'].append(res[1])
            hist['test_time'].append(test_time)
        except KeyError:
            hist['test_loss'] = [res[0]]
            hist['test_sparse_categorical_accuracy'] = [res[1]]
            hist['test_time'] = [test_time]

        print(f'loss: {res[0]}\nsca: {res[1]}\ntime: {test_time}s')
        
        if i >= 10: break
        
    hist['test_time'] = sum(hist['test_time']) / len(hist['test_time'])

    return hist

def plot_hist(hist, label):
    fig, splts = plt.subplots(2,3,sharey='row',sharex='col')
    y1 = hist['sparse_categorical_accuracy']
    x1 = np.arange(0,len(y1))+1
    splts[0,0].plot(x1,y1,'.-')
    
    y2 = hist['loss']
    x2 = np.arange(0,len(y2))+1
    splts[1,0].plot(x2,y2,'.-')
    
    y1 = hist['test_sparse_categorical_accuracy']
    x1 = np.arange(0,len(y1))+1
    splts[0,1].plot(x1,y1,'.-')
    
    y2 = hist['test_loss']
    x2 = np.arange(0,len(y2))+1
    splts[1,1].plot(x2,y2,'.-')
    
    y3 = hist['val_sparse_categorical_accuracy']
    x3 = np.arange(0,len(y3))+1
    splts[0,2].plot(x3,y3,'.-')
    
    y4 = hist['val_loss']
    x4 = np.arange(0,len(y4))+1
    splts[1,2].plot(x4,y4,'.-')

    splts[0,0].set_title('Train')
    splts[0,1].set_title('Test')
    splts[0,2].set_title('Validation')
    
    splts[0,0].set_xlabel('Epoch')
    splts[0,1].set_xlabel('Epoch')
    splts[0,2].set_xlabel('Epoch')

    #TODO: 
    #train model for 10 epochs
    #look into stripe 82 to improve query

    splts[0,0].set_ylabel('Sparse Categorical\nAccuracy')
    splts[1,0].set_ylabel('Loss')

    plt.show()

if __name__ == "__main__":
    try:
        with open('saved/resnet.hist','rb') as pkl:
            resnet_hist = load(pkl)
    except (FileNotFoundError,IOError):
        with open('../data/cifar-10/prep/cifar-10-test.pkl','rb') as pkl:
            test_imgs = load(pkl)
            test_labels = load(pkl)
        resnet_hist = test_model('saved\\resnet\\',
                                 RedshiftClassifierResNet,
                                 test_imgs, test_labels)
        with open('saved/resnet.hist', 'wb') as pkl:
            dump(resnet_hist, pkl)

    try:
        with open('saved/incep.hist','rb') as pkl:
            incep_hist = load(pkl)
    except (FileNotFoundError, IOError):
        try:
            test_imgs
            test_labels
        except NameError:
            with open('../data/cifar-10/prep/cifar-10-test.pkl','rb') as pkl:
                test_imgs = load(pkl)
                test_labels = load(pkl)
        incep_hist = test_model('saved\\incep\\',
                                RedshiftClassifierResNet,
                                test_imgs, test_labels)
        with open('saved/incep.hist', 'wb') as pkl:
            dump(incep_hist, pkl)


    plot_hist(resnet_hist, 'ResNet')
    plot_hist(incep_hist, 'Inception')
    