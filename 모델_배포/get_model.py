import tensorflow as tf
import tensorflow_hub as hub
import os

IMAGE_SHAPE = (224, 224)

def load_model():
    classifier_model = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2"

    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE + (3,))
    ])
    return classifier


def save_model():
    model = load_model()
    base_file_path = os.getcwd() + "/models/"

    file_path = base_file_path
    model.save(filepath=file_path, save_format='tf')
