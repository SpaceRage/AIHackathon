# %%
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelBinarizer
print("Tensorflow version: ", tf.__version__)

# %%
# unpickle all variables
with open('processed_dataset.pickle' , 'rb') as f:
    dataset = pickle.load(f)
    diag = pickle.load(f)
    localization = pickle.load(f)

# %%
mapping_dx = {
    'nevus': 0,
    'melanoma': 1,
    'keratosis-like': 0,
    'basal cell carcinoma': 0,
    'vascular lesion': 0,
    'dermatofibroma': 0,
    'lentigo': 0,
}
mapping_localization = {
    'lower extremity': 0,
    'torso': 1,
    'upper extremity': 2,
    'head/neck': 3,
    'palms/soles': 4,
    'oral/genital': 5,
    'unknown': 6,
    'hands/feet': 7
}

# %%
# create a new column with the mapping based on dx
dataset['dx'] = dataset['dx'].map(mapping_dx)
dataset['localization'] = dataset['localization'].map(mapping_localization)


# %%
# Define input and output shape
input_shape = (224, 224, 3)  # assuming RGB images of size 224x224
age_shape =  (1,)
sex_shape = (1,)
localization_shape = (1,)
num_classes = 1000  # assuming the number of output classes is 1
localization_shape = (1,)


age = np.array(dataset['age'], dtype='float16')
dx = np.array(dataset['dx'], dtype='float16')
images = np.array([img_to_array(img) for img in dataset['image']], dtype='float16')
images = np.repeat(images, 3, axis=3)
localizations = np.array(dataset['localization'], dtype='float16')
sex = dataset['sex'].to_numpy().astype('float16')
sex = sex.reshape(-1,1)
# rescale the images to 0-1
images = images / 255.0

tensor_age = tf.keras.Input(shape=age_shape, name='age', dtype='float16')
tensor_sex = tf.keras.Input(shape=sex_shape, name='sex',dtype='float16')
tensor_localization = tf.keras.Input(shape=localization_shape, name='localization',dtype='float16')
tensor_image = tf.keras.Input(shape=input_shape, name='image',dtype='float16')

base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')(tensor_image)
x = tf.keras.layers.Flatten()(base_model)

# Freeze the base model
base_model.trainable = False
# x = tf.keras.layers.concatenate([x, tensor_age,tensor_sex,tensor_localization])    
# Define fully connected layers
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
# output layer one neuron for each class
x = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)

# Define output layer
predictions = tf.keras.layers.Dense(num_classes)(x)

# Define the model
# model = tf.keras.models.Model(inputs=[tensor_image,tensor_age,tensor_sex,tensor_localization], outputs=predictions)
model = tf.keras.models.Model(inputs=[tensor_image,tensor_age,tensor_sex,tensor_localization], outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# %%
# split the data into train and test
train_data = images[:int(len(images)*0.8)]
train_age = age[:int(len(age)*0.8)]
train_sex = sex[:int(len(age)*0.8)]
train_localization = localizations[:int(len(localizations)*0.8)]
train_label = dx[:int(len(dx)*0.8)]

val_data = images[int(len(images)*0.8):]
val_age = age[int(len(images)*0.8):]
val_sex = sex[int(len(images)*0.8):]
val_localization = localizations[int(len(images)*0.8):]
val_labels = dx[int(len(images)*0.8):]

test_data = images[int(len(images)*0.8):]
test_age = age[int(len(images)*0.8):]
test_sex = sex[int(len(images)*0.8):]
test_localization = localizations[int(len(images)*0.8):]
test_labels = dx[int(len(images)*0.8):]

# %%
history = model.fit(
    [train_data,train_age,train_sex,train_localization], train_label,epochs=12, batch_size=32, validation_data=([val_data,val_age,val_sex,val_localization], val_labels)
)

# %%
test_loss, test_acc = model.evaluate([test_data,test_age,test_sex,test_localization], test_labels)

# %%
print('Test accuracy:', test_acc)

# %%
tf.keras.Model.save(model, 'test.h5')


