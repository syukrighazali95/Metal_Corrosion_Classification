import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from helper_functions import walk_through_dir, create_tensorboard_callback
from tensorflow.keras import layers

train_dir = "../Datasets/Classification datasets/train"
test_dir = "../Datasets/Classification datasets/test"

walk_through_dir("../Datasets/Classification datasets/")

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

input_shape = (224,224,3)

data_augmentation = tf.keras.Sequential([
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.2),
    preprocessing.Rescaling(scale=1./255)
], name="augmented_layer")

base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
base_model.trainable = False

inputs = layers.Input(shape=input_shape, name="input_layer")
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
outputs = layers.Dense(1, activation="sigmoid", name="output_layer")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.RMSprop(lr=0.02),
    metrics=["accuracy"]
)

initial_epoch = 0

fine_tune_epochs = initial_epoch + 7

history = model.fit(
    train_data,
    epochs=fine_tune_epochs,
    validation_data=test_data,
    validation_steps=int(0.25 * len(test_data)),
    initial_epoch=initial_epoch,
    callbacks=[create_tensorboard_callback(dir_name="transfer_learning", experiment_name="fine_tune_all_classes")]
    )

results = model.evaluate(test_data)

model.save("rust_binary_v2.h5")
