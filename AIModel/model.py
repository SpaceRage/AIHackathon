import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
print("Tensorflow version: ", tf.__version__)

# unpickle all variables
with open('processed_dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
    diag = pickle.load(f)
    localization = pickle.load(f)

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

# create a new column with the mapping based on dx
dataset['dx'] = dataset['dx'].map(mapping_dx)
dataset['localization'] = dataset['localization'].map(mapping_localization)


# Define input and output shape
input_shape = (224, 224, 3)  # assuming RGB images of size 224x224
age_shape =  (1,)
sex_shape = (1,)
localization_shape = (1,)
num_classes = 1  # assuming the number of output classes is 1

age = np.array(dataset['age'], dtype='float16')
dx = np.array(dataset['dx'], dtype='float16')
images = np.array([img_to_array(img) for img in dataset['image']], dtype='float16')
images = np.repeat(images, 3, axis=3)
localizations = np.array(dataset['localization'], dtype='float16')
sex = dataset['sex'].to_numpy().astype('float16')
sex = sex.reshape(-1, 1)
# rescale the images to 0-1
images = images / 255.0

# Define hyperparameters to search
def build_model(hp):
    tensor_age = tf.keras.Input(shape=age_shape, name='age', dtype='float16')
    tensor_sex = tf.keras.Input(shape=sex_shape, name='sex',dtype='float16')
    tensor_localization = tf.keras.Input(shape=localization_shape, name='localization',dtype='float16')
    tensor_image = tf.keras.Input(shape=input_shape, name='image',dtype='float16')

    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')(tensor_image)
    x = tf.keras.layers.Flatten()(base_model)

    # Freeze the base model
    base_model.trainable = False

    # Define fully connected layers
    for i in range(hp.Int('num_layers', 1, 6)):
        x = tf.keras.layers.concatenate([x, tensor_age, tensor_sex, tensor_localization])

        x = Dense(units=hp.Int('units_' + str(i), min_value=64, max_value=512, step=32), activation='relu')(x)
        # Add age, sex and localization as inputs to the fully connected layers
        
        x = tf.keras.layers.Dropout(hp.Float('dropout_' + str(i), 0, 0.5, step=0.1, default=0.25))(x)
        
        x = tf.keras.layers.BatchNormalization()(x)
        
        x= tf.keras.layers.Activation('relu')(x)
        
        x= tf.keras.layers.Dense(64, activation='relu')(x)
        
        x = tf.keras.layers.Dropout(hp.Float('dropout_' + str(i), 0, 0.5, step=0.1, default=0.25))(x)
        
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.Activation('relu')(x)
        
        
        # Define the output layer
        output = Dense(units=num_classes, activation='sigmoid', name='output')(x)

        # Define the model
    model = Model(inputs=[ tensor_image,tensor_age, tensor_sex, tensor_localization], outputs=output)

        # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    return model

tuner = RandomSearch(
build_model,
objective='val_accuracy',
max_trials=10,
executions_per_trial=1,
directory='tuner_results',
project_name='skin_cancer_diagnosis'
)
split = int(len(images)*0.8)
train_images, test_images = images[:split], images[split:]
train_dx, test_dx = dx[:split], dx[split:]
train_age, test_age = age[:split], age[split:]
train_sex, test_sex = sex[:split], sex[split:]
train_localizations, test_localizations = localizations[:split], localizations[split:]


tuner.search(x=[train_images,train_age, train_sex, train_localizations], y=train_dx,
epochs=10, validation_data=([test_images, test_age, test_sex, test_localizations], test_dx))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
from keras.utils import plot_model
plot_model(model, to_file='model.png')
model.fit(x=[images,age, sex, localizations], y=dx, epochs=10, validation_split=0.2)



model.save('skin_cancer_diagnosis_model.h5')

lb = LabelBinarizer()
lb.fit(dx)
with open('label_binarizer.pickle', 'wb') as f:
    pickle.dump(lb, f)