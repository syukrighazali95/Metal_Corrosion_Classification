import tensorflow as tf
import random
import matplotlib.pyplot as plt
from helper_functions import load_and_prep_image,load_keras_image
import os
from keras.preprocessing import image

train_dir = "D:/git/Metal_Corrosion_Classification/Datasets/Split datasets/train"
test_dir = "D:/git/Metal_Corrosion_Classification/Datasets/Split datasets/test"

IMG_SIZE = (224,224)

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    label_mode = "binary",
    image_size=IMG_SIZE,
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    label_mode="binary",
    image_size=IMG_SIZE
)

class_names = test_data.class_names

model = tf.keras.models.load_model("rust_binary_v2.h5")
print(model.summary())

model.evaluate(test_data)


img_path = 'D:/git/Metal_Corrosion_Classification/Datasets/Split datasets/test/rust/284.jpg'

load_keras_image(model, img_path, target_size=(224, 224))

plt.show()