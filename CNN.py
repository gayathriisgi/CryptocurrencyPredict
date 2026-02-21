import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

# Optional: reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================
# Load Data
# ==============================

with h5py.File('bitcoin2015to2017_close.h5', 'r') as hf:
    datas = hf['inputs'][:]      # FIXED (.value removed)
    labels = hf['outputs'][:]    # FIXED

output_file_name = 'bitcoin2015to2017_close_CNN_2_relu'

step_size = datas.shape[1]
batch_size = 8
nb_features = datas.shape[2]
epochs = 100

# ==============================
# Train / Validation Split
# ==============================

training_size = int(0.8 * datas.shape[0])

training_datas = datas[:training_size, :]
training_labels = labels[:training_size, :]

validation_datas = datas[training_size:, :]
validation_labels = labels[training_size:, :]

# ==============================
# Build Model
# ==============================

model = Sequential()

model.add(Conv1D(
    activation='relu',
    input_shape=(step_size, nb_features),
    strides=3,
    filters=8,
    kernel_size=20
))

model.add(Dropout(0.5))

model.add(Conv1D(
    strides=4,
    filters=nb_features,
    kernel_size=16
))

model.compile(loss='mse', optimizer='adam')

# Create weights folder if not exists
os.makedirs("weights", exist_ok=True)

# ==============================
# Train Model
# ==============================

model.fit(
    training_datas,
    training_labels,
    verbose=1,
    batch_size=batch_size,
    validation_data=(validation_datas, validation_labels),
    epochs=epochs,
    callbacks=[
        CSVLogger(output_file_name + '.csv', append=True),
        ModelCheckpoint(
            'weights/' + output_file_name + '-{epoch:02d}-{val_loss:.5f}.hdf5',
            monitor='val_loss',
            verbose=1,
            mode='min'
        )
    ]
)