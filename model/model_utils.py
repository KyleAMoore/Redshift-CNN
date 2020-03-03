from keras.utils import plot_model
from model import RedshiftClassifierResNet
from keras.models import load_model

def save_model_image(model, filename):
    plot_model(model, filename)

if __name__ == "__main__":
    resModel = RedshiftClassifierResNet((64, 64, 5), 32)
    save_model_image(resModel, 'redshift_resnet_model.png')