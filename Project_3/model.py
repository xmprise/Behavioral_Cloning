from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.optimizers import Adam
import os
import json
import numpy as np

from createdata import CreateData


# VGG16 using Batch Normalization
def get_model():

    model = Sequential()

    # Add a normalization layer
    model.add(Lambda(lambda x: x/127.5 - .5,
                     input_shape=(64, 64, 3),
                     output_shape=(64, 64, 3)))

    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, activation='elu', border_mode='same', name='block1_conv1'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 3, 3, activation='elu', border_mode='same', name='block1_conv2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Convolution2D(128, 3, 3, activation='elu', border_mode='same', name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Convolution2D(128, 3, 3, activation='elu', border_mode='same', name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    model.add(Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='block3_conv3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block4_conv2'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block4_conv3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block5_conv2'))
    model.add(BatchNormalization())
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block5_conv3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.add(Flatten(name='Flatten'))
    model.add(Dense(1024, activation='elu', name='fc1'))
    model.add(Dropout(0.5, name='fc1_dropout'))
    model.add(Dense(256, activation='elu', name='fc2'))
    model.add(Dropout(0.5, name='fc2_dropout'))
    model.add(Dense(128, activation='elu', name='fc3'))
    model.add(Dropout(0.5, name='fc3_dropout'))
    model.add(Dense(64, activation='elu', name='fc4'))
    model.add(Dropout(0.5, name='fc4_dropout'))
    model.add(Dense(32, activation='elu', name='fc5'))
    model.add(Dropout(0.5, name='fc5_dropout'))
    model.add(Dense(1, init='zero', name='output'))

    model.summary()
    model_json = model.to_json()
    with open('model.json', 'w') as f:
        f.write(model_json)

    return model

data = CreateData('./savedata/driving_log.csv')

X_train, X_valid, y_train, y_valid = data.load_data()

model = get_model()

batch_size = 120
samples_per_epoch = 20000
nb_epoch = 1

for layer in model.layers[0:2]:
    layer.trainable = True
for layer in model.layers[2:12]:
    layer.trainable = False
for layer in model.layers[12:]:
    layer.trainable = True

checkpoint = ModelCheckpoint("model-{epoch:03d}.h5", monitor='val_loss', verbose=0, save_best_only=True, mode='auto')


model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))

model.fit_generator(data.batch_data(X_train, y_train, batch_size), samples_per_epoch=160*batch_size, nb_epoch=2,
                    validation_data=data.batch_data(X_valid, y_valid, batch_size),
                    nb_val_samples=batch_size, callbacks=[checkpoint], verbose=1)