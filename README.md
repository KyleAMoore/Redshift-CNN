# Project Map
<pre>
Redshift-ResNet
├── <a href="model/README.md">model</a>
│   ├── <a href="model/model.py">model.py</a>
│   ├── <a href="model/eval_model.py">eval_model.py</a>
│   └── <a href="model/model_utils.py">model_utils.py</a>
├── <a href="preprocessing/README.md">preprocessing</a>
│   └── <a href="preprocessing/swarp/README.md">swarp</a>
├── <a href="train.py">train.py</a>
├── <a href="train-test.py">train-test.py</a>
└── <a href="display-galaxy.py">display-galaxy.py</a>
</pre>

# Redshift
This project seeks to automatically estimate the redshift value of astronomical bodies using pictures collected through astronomical sky surveys. This project in particular uses images from the Sloan Digital Sky Survey (SDSS).

## What is redshift?
[Redshift](https://en.wikipedia.org/wiki/Redshift) is a measurement of the difference between the apparant coloration of an astral body and its actual coloration. This disparity is related to the [doppler effect](https://en.wikipedia.org/wiki/Doppler_effect) and, as such, can be used to determine how quickly a body is moving away from the earth, which in turn can provide information on how far away that body is from the earth. Traditionally, this value is calculated using the body's measured light spectrum and the body's expected true light spectrum. In this project, the technique used by [Pasquet et. al.](https://arxiv.org/abs/1806.06607) to derive this value directly from astral images is replicated and extended.

# Requirements
In order to run this repository, you will need the following programs installed:

1. [SWarp](https://www.astromatic.net/software/swarp) - To resample the SDSS images
2. [Python 3](https://www.python.org/downloads/release/python-377/) - This repository was developed with python 3.7 and may not run correctly on earlier versions. In addition to the python libraries listed in [requirements.txt](requirements.txt), you will need to have [tensorflow](https://www.tensorflow.org/install) and all of its dependencies installed.

# What is in this repository?
## [train.py](train.py)
This script contains an API for training CNN models. The primary function of this API is:

`train_model(model, train_imgs, train_labels, arch_label, data_label, batch_size, epochs, num_rs_bins, max_rs_val, val_split, rot_chance, flip_chance):`

* model (required, keras.Model): The model architecture to be trained. Should be an already compiled keras model.
* train_imgs (required, numpy.array) - Images to train model on
* train_labels (required, numpy.array) - True desired values of outputs for each input
* arch_label (required, string) - Redshift values of training images
* data_label (required, string) - Name of architecture being tested. A folder with this name needs to exist in the location %Run_dir%/saved/
* batch_size (default: 32) - Number of images per training batch
* epochs (default: 10) - Number of times to train the model on the entire dataset
* num_rs_bins (default: 32) - Number of bins to use in model output
* max_rs_val (default: 3.5) - Maximum expected redshift value. Necessary for categorical conversion
* val_split (default: 0.15) - Percentage of samples to hold out for validation
* rot_chance (default: 0.4): Probability for each image to be individually flipped after each epoch (randomly chooses axis to flip across)
    * flip_chance (default: 0.2): Probability for each image to be individually rotated after each epoch (randomly chooses degree of rotation)

Sample execution:
```python
from pickle import load
from model.model import RedshiftClassifierResNet

with open('dataset.pkl','rb') as pkl:
    train_imgs = load(pkl)
    train_labels = load(pkl)

image_shape = (64,64,5)
num_classes = 32

model = RedshiftClassifierResNet(image_shape, num_classes)
 train_model(model, train_imgs, train_labels, 'incep', 'SDSS')
```

## [train-test.py](train-test.py)
This script is a ready-made example script for running the full training and testing pipeline using all default configurations. Three file locations need to be supplied in the indicated locations before the script can be executed. The first two are the locations of the pickle files that contain the training and testing data. Each of those files should contain two numpy arrays, with the first containing all of the input images, and the second containing the ground-truth output. The final location is where the results graph will be saved after testing has completed.

## [display-galaxy.py](display-galaxy.py)
This script provides the ability to display SDSS galaxy images in an approximation of their RGB coloration. It uses an implementation of the method described in [Bertin et. al.](http://www.aspbooks.org/publications/461/263.pdf). The primary functions of interest in the script are.

1. `BertinVisualizer(gamma, alpha, a, it, b, image_channels)`
    * gamma (default 2.4) - intensity scale when intensity is greater than it (default is for SDSS)
    * alpha (default 2.0) - saturation level of output images. Images are naturally desaturated, so values between 1 and 2 are recommended.
    * a (default 12.92) - intensity scale when intensity is less than it (default is for SDSS)
    * it (default 0.00304) - intensity threshold for switching transformation functions(default is for SDSS) 
    * b (default 0.055) - intensity scale when intensity is greater than it (default is for SDSS)
    * image_channels (default 5) - number of color channels in the image. For SDSS, these channels are ugriz (standard computer images are rgb)

2. `plot_images(images, redshifts, output_filename, num_bins, img_per_bin)`
    * images (required) - images to be converted to rgb
    * redshifts (required) - redshift values of bodies in images. Needed for binning and displaying by redshift
    * output_filename (required) - location to save the image plot
    * num_bins (default 7) - number of redshift bins (columns) to be displayed
    * img_per_bin (default 1) - number of images (rows) per bin

3. `process_img(image)`
    * image (required) - ugriz image to be converted to RGB

Sample execution:
```python
from pickle import load
import matplotlib.pyplot as plt

with open('images_file.pkl', 'rb') as pkl:
        images, redshifts = load(pkl).values()

rsvis = BertinVisualizer(alpha=1.2)

rsvis.plot_images(images, redshifts, 
                  'binned-galaxies-decomp.png',
                   num_bins=10, img_per_bin=8)

img_rgb = rsvis.process_img(images[0])
plt.imshow(img_rgb)
```
[Sample output](images/binned-galaxies-decomp.png)
