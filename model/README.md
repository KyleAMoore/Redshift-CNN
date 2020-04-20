# [model.py](model.py)
This module contains two CNN architectures for estimating redshift from images. They are detailed below

## ResNet
`RedshiftClassifierResNet(input_img_shape, num_redshift_classes, num_res_blocks, num_res_stacks, init_num_filters)`

* input_img_shape (required, tuple(int)) - Shape of the input images.
* num_redshift_classes (required, int) - Number of bins to split the resulting probability distribution function into.
* num_res_blocks (default 6) - Number of residual blocks per stack. Each block contains two convolutional layer and the output of the last convolutional layer is added to the input to the block to obtain the block's output
* num_res_stacks (default 4) - Number of stacks of residual blocks. Each stack (aside from the first) downsamples the images and increases the number of filters by a factor of 2
* init_num_filters (default 16) - Number of filters in the first convolutional layer. The number of filters in the final layer will be `init_num_filters * 2^(num_res_stacks - 1)`

Network topology with default settings can be viewed [here](images/ResNet-50.png)

## Inception
`RedshiftClassifierInception(input_img_shape, num_redshift_classes)`

* input_img_shape (required, tuple(int)) - shape of the input images.
* num_redshift_classes (required, int) - number of bins to split the resulting probability distribution function into.

Network topology can be viewed [here](images/Inception.png)

# [eval_models.py](eval_models.py)
This module provides an API for evaluating the performance of a trained redshift estimation model. The primary functions of interest are listed below.

1. `eval_models(models, test_data_path, model_directories, model_labels, output_path, max_val)`
    * models (required, list(keras.Model)) - Model objects to be evaluated (do not need to be trained)
    * test_data_path (required, string) - Location of the file containing the test data. Should be a pickle file containing two numpy.array objects (inputs followed by outputs)
    * model_directories (required, list(str)) - Directory containing the weights of the models to be evaluated. Model weights should be saved as `.hdf5` files. These directories should also contain the training history files with the file extension `.hist`
    * model_labels (required, list(str)) -Labels to be used for each model in the plotted results
    * output_path (required, str) - Location to save the generated results plot
    * max_rs_val - Maximum expected redshift value. Necessary for categorical conversion

    evaluates a collection of model and creates a plot comparing the models

2. `test_model(model, test_imgs, test_labels, directory, max_val)`
    * model (required, keras.Model) - Compiled and trained keras model
    * test_imgs (required, numpy.array) - Array of test images
    * test_labels (required, numpy.array) - Array of redshift values
    * directory (required, str) - Location of directory containing pre-trained model weights. This directory should also contain the training history file with the file extension `.hist`
    * max_rs_val - Maximum expected redshift value. Necessary for categorical conversion

    evaluates a single model. Returns a dictionary object containing the training loss and evaluation metric, as well as the test evaluation metrics at each epoch

The metrics used for evaluation are:

1. 'pred_bias' - Average bias of the model.
2. 'dev_MAD' - Deviation of the Median Absolute Deviation (MAD).
3. 'frac_outliers' - Fraction of predictions that were outliers (defined as having absolute bias >5x dev_MAD)
4. 'avg_crps' - Average Continuous Ranked Probability Score (CRPS)

Sample Execution:
```
image_shape = (64,64,5)
num_classes = 32

models = [
    RedshiftClassifierInception(image_shape, num_classes),
    RedshiftClassifierResNet(image_shape, num_classes)
]
directories = [
    'saved/incep/',
    'saved/resnet/'
]
labels = [
    'incep',
    'resnet'
]

eval_models(models, 'dataset.pkl', directories, labels, 'images/model_comp.png')
```
[Sample output](images/model_comp_SDSS.png)

# [model_utils.py](model_utils.py)
This module provides minor utilities for interacting with the models defined in [model.py](model.py). The provided functions are detailed below.

1. `save_model_image(model, filename)` - Saves an image of the keras graph for the model
    * model - Model to be displayed
    * filename - Location to save the model image