import numpy as np
from matplotlib import pyplot as plt
from model import RedshiftClassifierResNet, RedshiftClassifierInception
from pickle import dump, load
from glob import glob, iglob
from properscoring import crps_gaussian

def test_model(model, test_imgs, test_labels, directory):
    """Tests the model across all of its epochs
    
    Args:
        model (keras.Model): Compiled and trained keras model.
        test_imgs (numpy.array): Array of test images
        test_labels (numpy.array): Array of redshift values
        directory (str): Locations of directories containing pre-trained model weights.
    
    Returns:
        [type]: [description]
    """
    with open(glob(directory+'*.hist')[0],'rb') as pkl:
        hist = load(pkl)

    for weight_file in iglob(directory + '*.hdf5'):
        print('Testing model saved in location: ' + weight_file)
        model.load_weights(weight_file)
        results = redshift_evaluate(model, test_imgs, test_labels)
        
        for k, v in results.items():
            try:
                hist[k].append(v)
            except KeyError:
                hist[k] = [v]
    
    hist['train_sparse_categorical_accuracy'] = hist['sparse_categorical_accuracy']
    hist['train_loss'] = hist['loss']
    del(hist['sparse_categorical_accuracy'])
    del(hist['loss'])

    return hist

def redshift_evaluate(model, test_imgs, test_labels):
    """Evaluates the model using the metrics defined in https://arxiv.org/abs/1806.06607
    
    Args:
        model (keras.Model): Compiled and trained keras model.
        test_imgs (numpy.array): Array of test images
        test_labels (numpy.array): Array of redshift values
    
    Returns a dictionary with the following key-value pairs:
        'pred_bias' (float): Average bias of the model.
        'dev_MAD' (float): Deviation of the Median Absolute Deviation (MAD).
        'frac_outliers' (float): Fraction of predictions that were outliers (defined as
            having absolute bias >5x dev_MAD)
        'avg_crps' (float):  Average Continuous Ranked Probability Score (CRPS)
    """         
    pdfs = model.predict(test_imgs)
    
    step = model.max_val / model.num_classes
    bin_starts = np.arange(0,model.max_val,step)
    preds = np.sum((bin_starts+(step/2)) * pdfs, axis=1) # midpoints

    residuals = (preds - test_labels) / (test_labels + 1)
    pred_bias = np.average(residuals)
    dev_MAD = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
    frac_outliers = np.count_nonzero(np.abs(residuals) > (dev_MAD * 5)) / len(residuals)
    crps = np.average(crps_gaussian(preds, np.mean(preds), np.std(preds)))

    return {'pred_bias' : pred_bias,
            'dev_MAD' : dev_MAD,
            'frac_outliers' : frac_outliers,
            'avg_crps' : crps}

def plot_hist(histories, labels, output_filename):
    """Plots model evaluation metrics in the below format

        |----------------------------------| 
        |tr_loss|val_loss |devMAD|  crps   | 
        |----------------------------------| 
        |tr_sca |val_sca  |outlir|pred_bias| 
        |----------------------------------| 
    """
    fig, splts = plt.subplots(2,4, sharex='col', figsize=(16,8))
    
    for hist in histories:
        y = hist['train_loss']
        x = np.arange(1,len(y)+1)
        splts[0,0].plot(x,y,'.-')

        y = hist['train_sparse_categorical_accuracy']
        x = np.arange(1,len(y)+1)
        splts[1,0].plot(x,y,'.-')

        y = hist['val_loss']
        x = np.arange(1,len(y)+1)
        splts[0,1].plot(x,y,'.-')

        y = hist['val_sparse_categorical_accuracy']
        x = np.arange(1,len(y)+1)
        splts[1,1].plot(x,y,'.-')

        y = hist['dev_MAD']
        x = np.arange(1,len(y)+1)
        splts[0,2].plot(x,y,'.-')

        y = hist['frac_outliers']
        x = np.arange(1,len(y)+1)
        splts[1,2].plot(x,y,'.-')

        y = hist['avg_crps']
        x = np.arange(1,len(y)+1)
        splts[0,3].plot(x,y,'.-')

        y = hist['pred_bias']
        x = np.arange(1,len(y)+1)
        splts[1,3].plot(x,y,'.-')
                
    splts[0,0].set_title('train loss')
    splts[1,0].set_title('train sca')
    splts[0,1].set_title('val loss')
    splts[1,1].set_title('val_sca')
    splts[0,2].set_title('dev MAD')
    splts[1,2].set_title('fraction outliers')
    splts[0,3].set_title('average CRPS')
    splts[1,3].set_title('prediciton bias')

    splts[1,0].set_xlabel('Epoch')
    splts[1,1].set_xlabel('Epoch')
    splts[1,2].set_xlabel('Epoch')
    splts[1,3].set_xlabel('Epoch')
    
    splts[0,0].legend(labels)

    plt.savefig(output_filename, bbox_inches='tight')

def eval_models(models, test_data_path, model_directories, model_labels, output_path):
    """Evaluates and compares the models given on the same data
    
    Args:
        models (list: Keras.models): List of compiled keras models to be compared
        test_data_path (str): location of data to be used for testing. Should be a pickle file
            containing two numpy arrays (array of inputs followed by array of outputs)
        model_directories (list: str): locations of directories containing pre-trained model
            weights. The results of evaluation will also be stored in each of these directories
        model_labels (list: str): labels to be used for each model in the plotted results
        output_path (str): location to save the generated results plot
    """    
    with open(test_data_path, 'rb') as pkl:
        test_imgs = load(pkl)
        test_labels = load(pkl)

    histories : list = []
    for i,direc in enumerate(model_directories):
        try:
            with open(direc+'results.pkl','rb') as pkl:
                histories.append(load(pkl))

        except(FileNotFoundError, IOError):
            hist = test_model(models[i], test_imgs, test_labels, direc)
            with open(direc+'results.pkl', 'wb') as pkl:
                dump(hist, pkl)
            histories.append(hist)

    plot_hist(histories, model_labels, output_path)

def main(mode=0):
    """
        Runs one of a selection of preset evaluation sets.

        mode must be an integer in the range [0,1]

        1 - evaluates and compares a resnet model with an inception model
            with both using their default hyperparameters. (DEFAULT)

        2 - evaluates and compares a collection of resnet models of different
            topology. Compares a total of 15 models which combined have every
            combination of values of num_res_blocks and num_res_stacks in the
            ranges [4,8] and [3,5] respectively
    """
    image_shape = (64,64,5)
    num_classes = 32

    if mode == 0:
        models = [
            RedshiftClassifierInception(image_shape, num_classes),
            RedshiftClassifierResNet(image_shape, num_classes, num_res_blocks=6, num_res_stacks=4),
        ]
        directories = [
            'saved/incep/',
            'saved/resnet/'
        ]
        labels = [
            'incep',
            'resnet'
        ]

    elif mode == 1:
        models = []
        directories = []
        labels = []
        for num_blocks in range(4,9):
            for num_stacks in range(3,6):
                label = 'resnet-{b}-{s}'.format(b=num_blocks,s=num_stacks)
                directories.append('saved/'+label+'/')
                labels.append(label)
                models.append(RedshiftClassifierResNet(image_shape, num_classes,
                                                       num_res_blocks=num_blocks,
                                                       num_res_stacks=num_stacks))

    else:
        raise ValueError('Mode must be one of the integers 0 or 1 (default=0)')

    eval_models(models, '../data/SDSS/prep/sdss-test.pkl', directories, labels, 'images/model_comp.png')
