# -*- coding: utf-8 -*-
"""
Script for implementing comparisons to the proposed approach on the analysis of
building images for the damage level classification on a dataset labelled
from scratch using models trained from scratch as described in:

- Simone Saquella, Michele Scarpiniti, Wangyi Pu, Livio Pedone, Giulia Angelucci,
Michele Matteoni, Mattia Francioli, Stefano Pampanin, "Post-earthquake Damage
Assessment of Buildings Exploiting Data Fusion", in 2025 International Joint
Conference on Neural Networks (IJCNN 2025), Rome, Italy, June 30 - July 05, 2025.


@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np
import comparative_models as cm
import utils as ut
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import EarlyStopping



# Set main hyper-parameters
LR = 0.0001  # Learning rate
N_b = 32     # Batch size
N_e = 100    # Number of epochs for training the last layer


# Set data folder
data_folder = '../Data/'
save_folder = '../Models/'
result_folder = '../Results/'



# %% Load training set
training_set = data_folder + 'X_train.npy'
training_lab = data_folder + 'y_train.npy'


with tf.device('CPU/:0'):
    X = np.load(training_set)
    y = np.load(training_lab)

    np.random.seed(seed=42)
    idx = np.random.permutation(len(y))
    X = X[idx,:,:,:]
    y = y[idx]

    N_T = round(0.1*len(y))

    training_set = Dataset.from_tensor_slices((X[:-N_T],y[:-N_T]))
    training_set = training_set.batch(N_b)

    validation_set = Dataset.from_tensor_slices((X[-N_T:],y[-N_T:]))
    validation_set = validation_set.batch(N_b)



# %% Select the model

# Baseline CNN
net = cm.build_BaselineCNN(num_classes=4, LR=LR)
model_name = 'BaselineCNN'
# net.summary()


# AlexNet
# net = cm.build_AlexNet(num_classes=4, LR=LR)
# model_name = 'AlexNet'
# net.summary()


# GoogLeNet
# net = cm.build_GoogLeNet(num_classes=4, LR=LR)
# model_name = 'GoogLeNet'
# net.summary()


# Early stopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)



# %% Train the selected model

history = net.fit(training_set, epochs=N_e, validation_data=validation_set,
                   shuffle=True, callbacks=[early_stop])


# Save the trained model
save_file = save_folder + model_name + '.keras'
net.save(save_file)
# net.save(save_file, overwrite=True)


# Save the history
np.save(save_folder + model_name + '_history.npy', history)



# Plot curves
ut.plot_loss2(history.history, model_name)
ut.plot_accuracy2(history.history, model_name)


# Free memory after fine-tuning
del X, y, training_set, validation_set



# %% Testing

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, matthews_corrcoef


# Load test set
test_set = data_folder + 'X_test.npy'
test_lab = data_folder + 'y_test.npy'

Xt = np.load(test_set)
yt = np.load(test_lab)

yt_t = np.argmax(yt, axis=1)


# Load the trained model
# net = tf.keras.models.load_model(save_file)


# %% Evaluate the model
# results = net.evaluate(Xt, yt)
# print('Final Loss:', results[0])
# print('Overall Accuracy:', results[1])


# %% Evaluate the model output for test set
y_prob = net.predict(Xt)
y_pred = np.argmax(y_prob, axis=1)


# Evaluating the trained model
metrics = {}
metrics['acc'] = accuracy_score(yt_t, y_pred)
metrics['pre'] = precision_score(yt_t, y_pred, average='weighted')
metrics['rec'] = recall_score(yt_t, y_pred, average='weighted')
metrics['f1']  = f1_score(yt_t, y_pred, average='weighted')
metrics['auc'] = roc_auc_score(yt, y_prob, multi_class='ovo')
metrics['mcc'] = matthews_corrcoef(yt_t, y_pred)
metrics['report'] = classification_report(yt_t, y_pred, digits=4)



# Printing metrics
ut.print_metrics(metrics)


# Evaluating and Showing the confusion matrix (CM)
labels = ['None', 'Slight', 'Moderate', 'Heavy']

cm = confusion_matrix(yt_t, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
metrics['cm'] = cm

# Save the metrics
np.save(save_folder + model_name + '_metrics.npy', metrics)


# Save results on a text file
res_file = result_folder + 'Results_' + model_name + '.txt'
ut.print_file(res_file, metrics)
