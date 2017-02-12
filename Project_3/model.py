from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from createdata import CreateData


# VGG16
def get_model():

    model = Sequential()

    model.add(Lambda(lambda x: x/255 - .5,
                     input_shape=(64, 64, 3),
                     output_shape=(64, 64, 3)))

    model.add(Convolution2D(3, 1, 1, border_mode='same', name='conv_in'))
    model.add(Convolution2D(64, 3, 3, activation='elu', border_mode='same', name='conv_1'))
    model.add(Convolution2D(64, 3, 3, activation='elu', border_mode='same', name='conv_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool_1'))

    model.add(Convolution2D(128, 3, 3, activation='elu', border_mode='same', name='conv_3'))
    model.add(Convolution2D(128, 3, 3, activation='elu', border_mode='same', name='conv_4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool_2'))

    model.add(Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='conv_5'))
    model.add(Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='conv_6'))
    model.add(Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='conv_7'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool_3'))

    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='conv_8'))
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='conv_9'))
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='conv_10'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool_4'))

    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='conv_11'))
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='conv_12'))
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='conv_13'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool_5'))

    model.add(Flatten(name='Flatten'))
    model.add(Dense(512, activation='elu', name='fc1'))
    model.add(Dropout(0.5, name='fc1_dropout'))
    model.add(Dense(256, activation='elu', name='fc2'))
    model.add(Dropout(0.5, name='fc2_dropout'))
    model.add(Dense(64, activation='elu', name='fc3'))
    model.add(Dropout(0.5, name='fc3_dropout'))
    model.add(Dense(32, activation='elu', name='fc4'))
    model.add(Dropout(0.5, name='fc4_dropout'))
    model.add(Dense(1, init='zero', name='output'))

    model.summary()
    model_json = model.to_json()
    with open('model.json', 'w') as f:
        f.write(model_json)

    return model

data = CreateData('./savedata3/driving_log.csv')

X_train, X_valid, y_train, y_valid = data.load_data()

model = get_model()

batch_size = 120 # Track 2, 32

checkpoint = ModelCheckpoint("model-{epoch:03d}.h5", monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))

model.fit_generator(data.batch_data(X_train, y_train, batch_size), samples_per_epoch=160*batch_size, nb_epoch=5,
                    validation_data=data.batch_data(X_valid, y_valid, batch_size),
                    nb_val_samples=batch_size, callbacks=[checkpoint], verbose=1)
