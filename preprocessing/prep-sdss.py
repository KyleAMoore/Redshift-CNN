from pickle import dump, load
import numpy as np
from tqdm import tqdm
from random import shuffle

def load_data(file_name):
    with open(f'../data/sdss/{file_name}','rb') as pkl:
        data = load(pkl,encoding='bytes')

    return data

def save_data(*args, file_name='sdss-prepped.pkl'):
    with open(file_name, 'wb') as pkl:
        for obj in args:
            dump(obj,pkl)

def train_test_split(imgs, rs_vecs, test_size=0.2):
    indices = list(range(len(imgs)))
    shuffle(indices)
    test_ind  = indices[:int(len(imgs)*test_size)]
    train_ind = indices[int(len(imgs)*test_size):]

    return ((np.take(imgs, train_ind, axis=0), np.take(rs_vecs, train_ind)), 
            (np.take(imgs,  test_ind, axis=0), np.take(rs_vecs,  test_ind)))

if __name__ == "__main__":
    imgs, rs_vals = load_data('combined_dataset.pkl').values()
    train, test = train_test_split(imgs, rs_vals)

    save_data(*train, file_name='../data/sdss/prep/sdss-train.pkl')
    save_data(*test,  file_name='../data/sdss/prep/sdss-test.pkl')