from model.model import RedshiftClassifierResNet, RedshiftClassifierInception
from pickle import load, dump

train_data_file = <TRAIN_DATA_LOC>
test_data_file = <TEST_DATA_LOC>
results_file = <SAVE_LOC>

with open(train_data_file,'rb') as pkl:
    train_imgs = load(pkl)
    train_labels = load(pkl)

with open(test_data_file, 'rb') as pkl:
    test_imgs = load(pkl)
    test_labels = load(pkl)

image_shape = (64,64,5)
num_classes = 32
max_val = 0.4
epochs = 20

models = [
    RedshiftClassifierResNet(image_shape, num_classes, max_val),
    RedshiftClassifierInception(image_shape, num_classes, max_val)
]

model_labels = [
    'resnet',
    'incep'
]

results = dict()

for i, mod in enumerate(models):
    mod.train(train_imgs,
              train_labels,
              epochs=20,
              checkpoint_dir= 'checkpoints/' + model_labels[i])
    results[model_labels[i]] = mod.evaluate(test_imgs, test_labels)

with open(results_file, 'rb') as pkl:
    dump(results, pkl)