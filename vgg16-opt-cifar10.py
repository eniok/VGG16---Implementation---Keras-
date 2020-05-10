import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import sparse_softmax_cross_entropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K
from tensorflow.python.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
import os
import numpy as np

# get dataset
ds = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = ds.load_data()

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = ds.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

num_classes = 10

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

img_rows, img_cols = 32, 32
input_channels = 3
input_shape = (img_rows, img_cols, input_channels)
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(
        x_train.shape[0], input_channels, img_rows, img_cols)
    x_test = x_test.reshape(
        x_test.shape[0], input_channels, img_rows, img_cols)
    input_shape = (input_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(
        x_train.shape[0], img_rows, img_cols, input_channels)
    x_test = x_test.reshape(
        x_test.shape[0], img_rows, img_cols, input_channels)
    input_shape = (img_rows, img_cols, input_channels)


# architecture
input_layer = Input(input_shape)
# L1
conv1_1 = Conv2D(filters=64, kernel_size=(
    3, 3), padding="same", activation="relu")(input_layer)
conv1_2 = Conv2D(filters=64, kernel_size=(
    3, 3), padding="same", activation="relu")(conv1_1)
pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv1_2)
# L2
conv2_1 = Conv2D(filters=128, kernel_size=(
    3, 3), padding="same", activation="relu")(pool1)
conv2_2 = Conv2D(filters=128, kernel_size=(
    3, 3), padding="same", activation="relu")(conv2_1)
pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv2_2)
# L3
conv3_1 = Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu")(pool2)
conv3_2 = Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu")(conv3_1)
conv3_3 = Conv2D(filters=256, kernel_size=(
    3, 3), padding="same", activation="relu")(conv3_2)
pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv3_3)
# L4
conv4_1 = Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu")(pool3)
conv4_2 = Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu")(conv4_1)
conv4_3 = Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu")(conv4_2)
pool4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv4_3)
# L5
conv5_1 = Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu")(pool4)
conv5_2 = Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu")(conv5_1)
conv5_3 = Conv2D(filters=512, kernel_size=(
    3, 3), padding="same", activation="relu")(conv5_2)
pool5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(conv5_3)
# FC - OPTMIZED
flat = Flatten()(pool5)
fc1 = Dense(units=4096, activation="relu")(flat)
fc2 = Dense(units=4096, activation="relu")(fc1)
fc3 = Dense(units=1000, activation="relu")(fc2)
drop = Dropout(0.4)(fc3)
out = Dense(units=num_classes, activation="softmax")(drop)
# FC Original
# flat = Flatten()(pool5)
# fc1 = Dense(units=512, activation="relu")(flat)
# drop = Dropout(0.3)(fc1)
# fc2 = Dense(units=512, activation="relu")(drop)
# drop = Dropout(0.3)(fc2)
# out = Dense(units=num_classes, activation="softmax")(drop)

model = Model(inputs=input_layer, outputs=out, name='VGG16')


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 7:
        lr *= 1e-2
    elif epoch > 5:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % 'VGG16'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# choose training configs
sgd = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=sgd, loss=categorical_crossentropy,
              metrics=['accuracy'])
model.summary()


# train
hist = model.fit(x_train, y_train, batch_size=100, epochs=10,
                 shuffle=True, verbose=1, validation_split=0.1, callbacks=callbacks)

# test
model.evaluate(x_test, y_test, verbose=1)
