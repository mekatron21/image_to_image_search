# This helper function maps datasets into required dataset format used by base training, adversarial training, and all other training methods.

import tensorflow as tf
#from tensorflow.keras import models
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

# Load images, define train, test and dict keys
IMAGE_INPUT_NAME = 'image'
LABEL_INPUT_NAME = 'label'

def resize(dset, IMG_SIZE):  
    def sub_resize(image, label):
        #IMG_SIZE = 32
        """Resizes the images to (IMG_SIZE x IMG_SIZE) size."""
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        return image, label
    new_dataset = dset.map(sub_resize)
    return new_dataset

def format_example(dset, IMG_SIZE):
    def sub_format_example(image, label):
        image = tf.cast(image, tf.float32)
        image = image     #*1/255.0
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.image.grayscale_to_rgb(image)
        return image, label
    new_dataset = dset.map(sub_format_example)
    return new_dataset

def normalize(dset, IMG_SIZE):
    def sub_normalize(features):
        features[IMAGE_INPUT_NAME] = tf.cast(
          features[IMAGE_INPUT_NAME], dtype=tf.float32) / 255  #IMG_SIZE
        return features
    new_dataset = dset.map(sub_normalize)
    return new_dataset

def examples_to_tuples(dset, IMG_SIZE):
    def to_tuples(features):
        return features[IMAGE_INPUT_NAME], features[LABEL_INPUT_NAME]
    new_dataset = dset.map(to_tuples)
    return new_dataset

def examples_to_dict(dset, IMG_SIZE):
    def to_dict(image, label):
        return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}
    new_dataset = dset.map(to_dict)
    return new_dataset

def normalize2(dset, IMG_SIZE):
    def sub_normalize(features):
        features[IMAGE_INPUT_NAME] = tf.cast(
          features[IMAGE_INPUT_NAME], dtype=tf.float32)   #IMG_SIZE
        return features
    new_dataset = dset.map(sub_normalize)
    return new_dataset

def normalize3(dset, IMG_SIZE):
    def sub_normalize(features):
        features[IMAGE_INPUT_NAME] = tf.cast(
          features[IMAGE_INPUT_NAME], dtype=tf.float32)  * 255 #IMG_SIZE
        return features
    new_dataset = dset.map(sub_normalize)
    return new_dataset



def main_map_fun(dset,funn, IMG_SIZE):
    new_dataset = funn(dset, IMG_SIZE)
    return new_dataset
 

