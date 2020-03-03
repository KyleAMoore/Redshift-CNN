import numpy as np
from matplotlib import pyplot as plt
from model import RedshiftClassifierResNet, RedshiftClassifierInception
from pickle import dump, load
from glob import glob, iglob
from time import time


def test_model(directory, model, test_imgs, test_labels):
    with open(glob(directory+'*.hist')[0],'rb') as pkl:
        hist = load(pkl)

    for weight_file in iglob(directory + '*weights.*.hdf5'):
        print('Testing model saved in location: ' + weight_file)
        model.load_weights(weight_file)
        start = time()
        res = model.evaluate(test_imgs, test_labels)
        test_time = time() - start
        try:
            hist['test_loss'].append(res[0])
            hist['test_sparse_categorical_accuracy'].append(res[1])
            hist['test_time'].append(test_time)
        except KeyError:
            hist['test_loss'] = [res[0]]
            hist['test_sparse_categorical_accuracy'] = [res[1]]
            hist['test_time'] = [test_time]

        print('loss: ' + str(res[0]))
        print('sparse categorical accuracy: '  + str(res[1]))
        print('time: ' + str(test_time)+'s')
    
    hist['test_time'] = sum(hist['test_time']) / len(hist['test_time'])
    hist['train_sparse_categorical_accuracy'] = hist['sparse_categorical_accuracy']
    hist['train_loss'] = hist['loss']
    del(hist['sparse_categorical_accuracy'])
    del(hist['loss'])

    return hist

def plot_hist(histories, labels, metrics, sets, output_filename):
    fig, splts = plt.subplots(2,3,
                              sharey='row',
                              sharex='col',
                              figsize=(len(sets)*4,len(metrics)*4))
    
    for hist in histories:
        for met_i, met in enumerate(metrics):
            for set_i, set in enumerate(sets):
                y = hist[set + '_' + met]
                x = np.arange(0,len(y))+1
                splts[met_i,set_i].plot(x,y,'.-')
                
    for set_i, set in enumerate(sets):
        splts[0,set_i].set_title(set)
        splts[1,set_i].set_xlabel('Epoch')

    for met_i, met in enumerate(metrics):
        splts[met_i,0].set_ylabel(met)
    
    splts[0,0].legend(labels)

    # plt.show()
    plt.savefig(output_filename, bbox_inches='tight')

if __name__ == "__main__":
    with open('../data/cifar-10/prep/cifar-10-test.pkl','rb') as pkl:
            test_imgs = load(pkl)
            test_labels = load(pkl)
            image_shape = (32,32,3)
            num_classes = 10

    test_imgs = test_imgs
    test_labels = test_labels

    try:
        with open('saved/resnet/resnet_tested.pkl','rb') as pkl:
            resnet_hist = load(pkl)
    except(FileNotFoundError,IOError):
        resnet_hist = test_model('saved/resnet/',
                                 RedshiftClassifierResNet(image_shape, num_classes),
                                 test_imgs, test_labels)
        with open('saved/resnet/resnet_tested.pkl', 'wb') as pkl:
            dump(resnet_hist, pkl)

    try:
        with open('saved/incep/incep_tested.pkl','rb') as pkl:
            incep_hist = load(pkl)
    except(FileNotFoundError, IOError):
        incep_hist = test_model('saved/incep/',
                                RedshiftClassifierInception(image_shape, num_classes),
                                test_imgs, test_labels)
        with open('saved/incep/incep_tested.pkl', 'wb') as pkl:
            dump(incep_hist, pkl)

    plot_hist(histories=[resnet_hist, incep_hist],
              labels=['ResNet50', 'Inception'],
              metrics=['sparse_categorical_accuracy','loss'],
              sets=['train', 'val', 'test'],
              output_filename='images/model_comp_cifar10.png')
    