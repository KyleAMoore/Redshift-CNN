import numpy as np
from matplotlib import pyplot as plt
from pickle import dump, load

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

def eval_models(models, test_data_path, model_directories, model_labels, output_path, max_val):
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
            hist = test_model(models[i], test_imgs, test_labels, direc, max_val)
            with open(direc+'results.pkl', 'wb') as pkl:
                dump(hist, pkl)
            histories.append(hist)

    plot_hist(histories, model_labels, output_path)