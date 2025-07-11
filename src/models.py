# -*- coding: utf-8 -*-
"""
Definition of the EfficientNet-B0 model and related fusion strategies to be
used in analysing building images for the damage level classification on a
dataset labelled from scratch as described in:

- Simone Saquella, Michele Scarpiniti, Wangyi Pu, Livio Pedone, Giulia Angelucci,
Michele Matteoni, Mattia Francioli, Stefano Pampanin, "Post-earthquake Damage
Assessment of Buildings Exploiting Data Fusion", in 2025 International Joint
Conference on Neural Networks (IJCNN 2025), Rome, Italy, June 30 - July 05, 2025.


@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.applications import EfficientNetB0




# Function for defining a new EfficientNet-B0 model
def build_model(num_classes, LR=0.001):
    inputs = Input(shape=(224, 224, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



# Function for defining a new EfficientNet-B0 model for Early fusion
def build_EF_model(num_classes, LR=0.001):
    inputs = Input(shape=(224, 448, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



# Function for defining a new fusion strategy with EfficientNet-B0 model
def build_IF_model(num_classes, LR=0.001):
    inputs1 = Input(shape=(224, 224, 3))
    model1 = EfficientNetB0(include_top=False, input_tensor=inputs1, weights="imagenet")
    for l in model1.layers:
        # l._name = l.name + '_1'
        l.name = l.name + '_1'

    inputs2 = Input(shape=(224, 224, 3))
    model2 = EfficientNetB0(include_top=False, input_tensor=inputs2, weights="imagenet")
    for l in model2.layers:
        # l._name = l.name + '_2'
        l.name = l.name + '_2'

    # Freeze the pretrained weights
    model1.trainable = False
    model2.trainable = False

    # Rebuild top
    x1 = GlobalAveragePooling2D(name="avg_pool_1")(model1.output)
    x1 = BatchNormalization()(x1)

    x2 = GlobalAveragePooling2D(name="avg_pool_2")(model2.output)
    x2 = BatchNormalization()(x2)

    x = concatenate([x1, x2], axis=1)

    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model([inputs1, inputs2], outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



# Function for unfreezing layers in EfficientNet-B0
def unfreeze_model(model, N_unfreeze=20, LR=1e-5):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-N_unfreeze:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )




# Function for unfreezing all layers in EfficientNet-B0
def unfreeze_all_model(model, LR=1e-5):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=LR)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
