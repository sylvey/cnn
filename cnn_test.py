import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils
from matplotlib import image
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
import warnings
import matplotlib.pyplot as plt

def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image.astype('uint8'), cmap='binary')
    plt.show()

# load the image
img = load_img('1.jpeg', target_size=(200, 200))
# report details about the image
print(type(img))
print(img.format)
print(img.mode)
print(img.size)

img_array = img_to_array(img)

x_Train4D = np.array([img_array])
print(x_Train4D.shape)

