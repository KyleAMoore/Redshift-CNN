from model.model import RedshiftClassifierResNet, RedshiftClassifierInception
from model.eval_model import eval_models
from train import train_model
from pickle import load

train_data_file = <TRAIN_DATA_LOC>
test_data_file = <TEST_DATA_LOC>
results_file = <SAVE_LOC>

with open(train_data_file,'rb') as pkl:
    train_imgs = load(pkl)
    train_labels = load(pkl)

image_shape = (64,64,5)
num_classes = 32
epochs = 20
max_val=0.4

models = [
    RedshiftClassifierResNet(image_shape, num_classes),
    RedshiftClassifierInception(image_shape, num_classes)
]

model_labels = [
    'resnet',
    'incep'
]

data_label = 'SDSS'

for i, mod in enumerate(models):
    train_model(mod,
                train_imgs,
                train_labels,
                model_labels[i],
                data_label,
                epochs=epochs,
                max_rs_val=max_val)

model_dirs = ['model/saved/'+lab+'/' for lab in model_labels]

eval_models(models,
            test_data_file,
            model_dirs,
            model_labels,
            results_file,
            max_val)