# %%
import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers


# %%
IMAGE_SIZE = 128
BATCH_SIZE = 4
NUM_CLASSES = 20
DATA_DIR = "data"
NUM_TRAIN_IMAGES = 80
NUM_VAL_IMAGES = 20

#TRAIN X
#TRAIN Y
train_images = sorted(glob(os.path.join(DATA_DIR, "images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "masks/*")))[:NUM_TRAIN_IMAGES]

#TEST X
#TEST Y
val_images = sorted(glob(os.path.join(DATA_DIR, "images/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]


#image reader
def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    #mask
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    #raw image
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127.5 - 1
    return image

#func for map: image reader -> dataset
def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


# %%
#genereate datasets

#we use from_tensor_slices as we have our X and Y in one dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
train_dataset = train_dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

val_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
val_dataset = val_dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)


# %%
def display(display_list):
  plt.figure(figsize=(10, 10))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

# %%
for images, masks in train_dataset.take(2):
  sample_image, sample_mask = images[0], masks[0]
  display([sample_image, sample_mask])


# %%
from unittest import skip
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from keras.models import Model

def convolution_block(input, num_filters):
    #          num filter  kernel size
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    return x

def encoder_step(input, num_filters):
    skip_connection_output = convolution_block(input, num_filters)
    encoder_step_output = MaxPool2D((2,2))(skip_connection_output)
    
    return skip_connection_output, encoder_step_output

def decoder_step(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = convolution_block(x, num_filters)
    
    return x

def build_unet(input_shape):
    input = Input(input_shape)
    
    """Encoder"""
    #                       256
    skip_connection_output1, encoder_step_output1 = encoder_step(input, 64)
    #                       128
    skip_connection_output2, encoder_step_output2 = encoder_step(encoder_step_output1, 128)
    #                       64
    skip_connection_output3, encoder_step_output3 = encoder_step(encoder_step_output2, 256)
    #                       32
    skip_connection_output4, encoder_step_output4 = encoder_step(encoder_step_output3, 512)    

    """Bridge"""
    #                       32
    bridge1 = convolution_block(encoder_step_output4, 1024)
    
    """Decoder"""
    #   32  ->   64
    decoder_step_output1 = decoder_step(bridge1, skip_connection_output4, 512)
    decoder_step_output2 = decoder_step(decoder_step_output1, skip_connection_output3, 256)
    decoder_step_output3 = decoder_step(decoder_step_output2, skip_connection_output2, 128)
    decoder_step_output4 = decoder_step(decoder_step_output3, skip_connection_output1, 64)
    
    """Output"""
    #            output channels  srides
    output = Conv2D(NUM_CLASSES, (1,1), padding="same", activation="sigmoid")(decoder_step_output4)
    
    model = Model(input, output, name="u-net")
    
    return model
    

# %%
input_shape = (128,128, 3)
model = build_unet(input_shape)
model.summary()

# %%
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"],
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=25)

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.show()


