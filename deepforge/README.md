# DeepForge

DeepForge is a graphical framework for the designing of deep neural networks in a more user-friendly manner than python programming. Information on using Deepforge can be found on the [DeepForge website](http://deepforge.org/) or in the public [DeepForge repository](https://github.com/deepforge-dev/deepforge) on GitHub.

# Redshift-DeepForge

Included in this directory is a DeepForge project that implements the training and testing pipeline at the root of this repository. It can be loaded by importing [redshift.webgmex](redshift.webgmex) into your DeepForge system as a new project. The project includes three pipelines:

1. Test-Pretrained - For evaluating a pretrained keras model that was saved using the [keras.Model.save](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) function or created and saved by another DeepForge pipeline.
2. Train-Test-Single - Trains and evaluates a single CNN model.
3. Train-Test-Compare - Trains, evaluates, and compares two models (the two included models by default).

All three pipelines will require the importing of training and/or testing datasets. Each dataset should be a pair of pickle files containing numpy arrays. The first array should contain the input images and be of shape `(n, 64, 64, 5)` where n is the number of images. The second array should contain the redshift values of the images in the first array and should be of shape `(n,)`

This project also comes with two models implemented directly in DeepForge, Inception and ResNet-50. These are implemented exactly as they are in [model/model.py](../model/model.py). The ResNet-50 model specifically is implemented using the default inputs to the `RedshiftClassificationResNet` class (6 blocks per residual stack and 4 residual stacks.)
