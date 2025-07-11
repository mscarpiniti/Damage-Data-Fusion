# -*- coding: utf-8 -*-
"""
Definition of some well-known models used to provide comparisons in analysing
building images for the damage level classification on a dataset labelled from
scratch as described in:

- Simone Saquella, Michele Scarpiniti, Wangyi Pu, Livio Pedone, Giulia Angelucci,
Michele Matteoni, Mattia Francioli, Stefano Pampanin, "Post-earthquake Damage
Assessment of Buildings Exploiting Data Fusion", in 2025 International Joint
Conference on Neural Networks (IJCNN 2025), Rome, Italy, June 30 - July 05, 2025.


@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D
from tensorflow.keras.layers import Input, Flatten, concatenate
from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import DenseNet201





# Function for unfreezing all layers in pre-trained models --------------------
def unfreeze_model(model, LR=1e-5):
    # We unfreeze all layers while leaving BatchNorm layers frozen
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )




# Function for defining a new MobileNet-V2 ------------------------------------
def build_MobileNetV2(num_classes, LR=0.001):
    inputs = Input(shape=(224, 224, 3))
    model = MobileNetV2(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = Model(inputs, outputs, name="MobileNet-V2")
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



# Function for defining the DenseNet201 ---------------------------------------
def build_DenseNet201(num_classes, LR=0.001):
    base_model = DenseNet201(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(224, 224, 3),
        include_top=False,
        classes=num_classes
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = Input(shape=(224, 224, 3))

    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)  # Regularize with dropout
    outputs = Dense(num_classes)(x)

    # Compile
    model = Model(inputs, outputs, name="DenseNet201")
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # model.summary(show_trainable=True)

    return model




# Function for defining the Baseline CNN --------------------------------------
def build_BaselineCNN(num_classes, LR=0.001):
    inputs = Input(shape=(224, 224, 3))

    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="BaselineCNN")

    # model.summary()

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



# Function for defining the AlexNet -------------------------------------------
def build_AlexNet(num_classes, LR=0.001):
    inputs = Input(shape=(224, 224, 3))

    x = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
    x = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
    x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="AlexNet")

    # Display the model's architecture
    # model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model



# Functions for defining the GoogLeNet ----------------------------------------

# Define the "inception" module
def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):

    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output



# Function for defining the GoogLeNet
def build_GoogLeNet(num_classes, LR=0.0001):

    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)


    input_layer = Input(shape=(224, 224, 3))

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')

    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64,
                         name='inception_3b')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         name='inception_4a')


    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(num_classes, activation='softmax', name='auxilliary_output_1')(x1)

    x = inception_module(x,
                         filters_1x1=160,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4b')

    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4c')

    x = inception_module(x,
                         filters_1x1=112,
                         filters_3x3_reduce=144,
                         filters_3x3=288,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4d')


    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(num_classes, activation='softmax', name='auxilliary_output_2')(x2)

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_4e')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5a')

    x = inception_module(x,
                         filters_1x1=384,
                         filters_3x3_reduce=192,
                         filters_3x3=384,
                         filters_5x5_reduce=48,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5b')

    x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

    x = Dropout(0.4)(x)

    x = Dense(num_classes, activation='softmax', name='output')(x)


    model = Model(input_layer, [x, x1, x2], name='GoogLeNet')
    # model.summary()

    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                  loss_weights=[1, 0.3, 0.3],
                  optimizer=keras.optimizers.Adam(learning_rate=LR),
                  metrics=['accuracy'])

    return model
