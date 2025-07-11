# -*- coding: utf-8 -*-
"""
Definition of useful functions for implementing the EfficientNet-B0 model and
related fusion strategies to be used in analysing building images for the damage
level classification on a dataset labelled from scratch as described in:

- Simone Saquella, Michele Scarpiniti, Wangyi Pu, Livio Pedone, Giulia Angelucci,
Michele Matteoni, Mattia Francioli, Stefano Pampanin, "Post-earthquake Damage
Assessment of Buildings Exploiting Data Fusion", in 2025 International Joint
Conference on Neural Networks (IJCNN 2025), Rome, Italy, June 30 - July 05, 2025.


@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

# import numpy as np
import matplotlib.pyplot as plt



# Function to flat a nested list ----------------------------------------------
def flat_list(L):
    flatList = [el for inList in L for el in inList]

    return flatList



# Function to plot the training and validation accuracy of the model ----------
def plot_accuracy(history, fusion):
    L1 = flat_list(history['accuracy'])
    L2 = flat_list(history['val_accuracy'])
    L_e = max(len(L1), len(L2))
    ep = range(1, L_e+1)

    plt.figure(figsize=(8, 6))
    plt.plot(ep, L1, linewidth=2, label='Train Accuracy')
    plt.plot(ep, L2, linewidth=2, label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy - {fusion}', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()



# Function to plot the training and validation loss of the model --------------
def plot_loss(history, fusion):
    L1 = flat_list(history['loss'])
    L2 = flat_list(history['val_loss'])
    L_e = max(len(L1), len(L2))
    ep = range(1, L_e+1)

    plt.figure(figsize=(8, 6))
    plt.plot(ep, L1, linewidth=2, label='Train Loss')
    plt.plot(ep, L2, linewidth=2, label='Validation Loss')
    plt.title(f'Training and Validation Loss - {fusion}', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()



# Function to plot the training and validation accuracy of the model ----------
def plot_accuracy2(history, fusion):
    L1 = history['accuracy']
    L2 = history['val_accuracy']
    L_e = max(len(L1), len(L2))
    ep = range(1, L_e+1)

    plt.figure(figsize=(8, 6))
    plt.plot(ep, L1, linewidth=2, label='Train Accuracy')
    plt.plot(ep, L2, linewidth=2, label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy - {fusion}', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()



# Function to plot the training and validation loss of the model --------------
def plot_loss2(history, fusion):
    L1 = history['loss']
    L2 = history['val_loss']
    L_e = max(len(L1), len(L2))
    ep = range(1, L_e+1)

    plt.figure(figsize=(8, 6))
    plt.plot(ep, L1, linewidth=2, label='Train Loss')
    plt.plot(ep, L2, linewidth=2, label='Validation Loss')
    plt.title(f'Training and Validation Loss - {fusion}', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()



# Function for printing metrics -----------------------------------------------
def print_metrics(metrics):
    print("Overall accuracy: {}%".format(round(100*metrics['acc'],2)))
    print("Precision: {}".format(round(metrics['pre'],4)))
    print("Recall: {}".format(round(metrics['rec'],4)))
    print("F1-score: {}".format(round(metrics['f1'],4)))
    print("AUC: {}".format(round(metrics['auc'],4)))
    print("MCC: {}".format(round(metrics['auc'],4)))
    print(" ", end='\n')
    print("Complete report: ", end='\n')
    print(metrics['report'])
    print(" ", end='\n')



# Function for saving results on a text file ----------------------------------
def print_file(res_file, metrics):
    with open(res_file, 'a') as results:  # save the results in a .txt file
          results.write('-------------------------------------------------------\n')
          results.write('Acc: %s\n' % round(100*metrics['acc'],2))
          results.write('Pre: %s\n' % round(metrics['pre'],4))
          results.write('Rec: %s\n' % round(metrics['rec'],4))
          results.write('F1: %s\n' % round(metrics['f1'],4))
          results.write('AUC: %s\n' % round(metrics['auc'],4))
          results.write('MCC: %s\n\n' % round(metrics['mcc'],4))
          results.write(metrics['report'])
          results.write('\n\n')
