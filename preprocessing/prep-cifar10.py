from pickle import dump, load
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_data(file_name):
    with open(f'../data/cifar-10/{file_name}','rb') as pkl:
        data = load(pkl,encoding='bytes')

    return data

def save_data(*args, file_name='cifar-10-prepped.pkl'):
    with open(file_name, 'wb') as pkl:
        for obj in args:
            dump(obj,pkl)

def process_img(img_arr):
    ret_img = conv_to_3D(img_arr)

    return ret_img

def conv_to_3D(img_arr):
    """
    Converts the images from original 2D format to more traditional 3D format

    Initial format is a 3072-length array where the first 1024 entries correspond
    to the red channel, the next 1024 to the green channel, and the remaining the
    blue channel. Within each channel group, the 32*32 images are listed in row major
    order such that the first 32 entries are the channel values for the first row
    of the image.

    Input: A numpy array of shape 3072*1

    Returns: A numpy array of shape 32*32*3 that represent the original image
    """
    r,g,b = np.split(img_arr,3)
    img = np.array([[r[i],g[i],b[i]] for i in range(len(r))])
    return np.split(img,32)
    
def prep_train_data():
    train_data: list = []
    train_labels: list = []
    for i in range(1,6):
        data = load_data(f'data_batch_{i}')
        train_data += [process_img(img) for img in tqdm(data[b'data'],desc=f'batch {i}')]
        train_labels += data[b'labels']
    return np.array(train_data), np.array(train_labels)

def prep_test_data():
    data = load_data('test_batch')
    test_data = np.array([process_img(img) for img in tqdm(data[b'data'],desc='test batch')])
    test_labels = data[b'labels']

    return np.array(test_data), np.array(test_labels)

if __name__ == "__main__":
    save_data(*prep_train_data(), file_name='../data/cifar-10/prep/cifar-10-train.pkl')
    save_data(*prep_test_data(),  file_name='../data/cifar-10/prep/cifar-10-test.pkl')