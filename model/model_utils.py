from keras.utils import plot_model
from model import RedshiftClassifierResNet
from keras.models import load_model

def save_model_image(model, filename):
    plot_model(model, filename, show_shapes=True)
