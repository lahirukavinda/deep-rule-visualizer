"""
Please cite:
Angelov, P., & Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks.
"""

###############################################################################
import pandas as pd

from xDNN.xDNN_class import *
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time


def xDNN_run(data_set):

    # Load the files, including features, images and labels.
    X_train_file_path = rf'features_{data_set}/data_df_X_train.csv'
    y_train_file_path = rf'features_{data_set}/data_df_y_train.csv'
    X_test_file_path = rf'features_{data_set}/data_df_X_test.csv'
    y_test_file_path = rf'features_{data_set}/data_df_y_test.csv'

    X_train = genfromtxt(X_train_file_path, delimiter=',')
    y_train = pd.read_csv(y_train_file_path, delimiter=';',header=None)
    X_test = genfromtxt(X_test_file_path, delimiter=',')
    y_test = pd.read_csv(y_test_file_path, delimiter=';',header=None)


    # Print the shape of the data

    print ("###################### Data Loaded ######################")
    print("Data Shape:   ")
    print("X train: ",X_train.shape)
    print("Y train: ",y_train.shape)
    print("X test: ",X_test.shape)
    print("Y test: ",y_test.shape)


    pd_y_train_labels = y_train[1]
    pd_y_train_images = y_train[0]

    pd_y_test_labels = y_test[1]
    pd_y_test_images = y_test[0]



    # Convert Pandas to Numpy
    y_train_labels = pd_y_train_labels.to_numpy()
    y_train_images = pd_y_train_images.to_numpy()

    y_test_labels = pd_y_test_labels.to_numpy()
    y_test_images = pd_y_test_images.to_numpy()



    # Model Learning
    Input1 = {}

    Input1['Images'] = y_train_images
    Input1['Features'] = X_train
    Input1['Labels'] = y_train_labels

    Mode1 = 'Learning'

    start = time.time()
    Output1 = xDNN(Input1,Mode1)

    end = time.time()

    print ("###################### Model Trained ####################")

    print("Time: ",round(end - start,2), "seconds")
    ###############################################################################

    # Load the files, including features, images and labels for the validation mode

    Input2 = {}
    Input2['xDNNParms'] = Output1['xDNNParms']

    # Validate testing dataset
    # Input2['Images'] = y_test_images
    # Input2['Features'] = X_test
    # Input2['Labels'] = y_test_labels

    # Validate only the first image of testing dataset
    Input2['Images'] = y_test_images[0:1]  # y_test_images[0].reshape(1,-1)
    Input2['Features'] = X_test[0:1]
    Input2['Labels'] = y_test_labels[0:1]
    print(f"\nValidate for image : {Input2['Images'][0]}\nOriginal label : {Input2['Labels'][0]}")



    startValidation = time.time()
    Mode2 = 'Validation'
    Output2 = xDNN(Input2,Mode2)
    endValidation = time.time()

    # print ("###################### Results ##########################")
    #
    #
    # # Elapsed Time
    # print("Time: ", round(endValidation - startValidation,2), "seconds")
    # # accuracy: (tp + tn) / (p + n)
    # accuracy = accuracy_score(y_test_labels , Output2['EstLabs'])
    # print('Accuracy: %f' % accuracy)
    # # precision tp / (tp + fp)
    # precision = precision_score(y_test_labels , Output2['EstLabs'], average='micro')
    # print('Precision: %f' % precision)
    # # recall: tp / (tp + fn)
    # recall = recall_score(y_test_labels , Output2['EstLabs'],average='micro')
    # print('Recall: %f' % recall)
    # # f1: 2 tp / (2 tp + fp + fn)
    # f1 = f1_score(y_test_labels , Output2['EstLabs'], average='micro')
    # print('F1 score: %f' % f1)
    # # kappa
    # kappa = cohen_kappa_score(y_test_labels , Output2['EstLabs'])
    # print('Cohens kappa: %f' % kappa)
    #
    # # confusion matrix
    # matrix = confusion_matrix(y_test_labels , Output2['EstLabs'])
    # print("Confusion Matrix: ",matrix)