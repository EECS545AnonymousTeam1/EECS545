# -*- coding: utf-8 -*-
"""TestingLSA.ipynb

Automatically generated by Colaboratory.

"""

import numpy as np
import csv
import string
import sys
import re
import spacy
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import matplotlib
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV, cross_val_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from getpass import getpass
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import math

# Copy the files from the eecs-545-project Gitlab repository
!rm -r eecs-545-project
!rm -r eecs_545_project

user = getpass('Gitlab user')
password = getpass('Gitlab password')

os.environ['GITLAB_AUTH'] = user + ':' + password

from google.colab import drive
drive.mount('/content/drive')

# Load PyTorch test data and labels
torch_test_data = torch.load('/content/drive/My Drive/bert_articles_for_testing.pt')
torch_test_labels = torch.zeros(1500, dtype=torch.int64, device='cuda:0')
for i in range(15):
  torch_test_labels[(100 * i):(100 * i + 100)] = i
print(torch_test_labels)

print(torch_test_data.shape)

# Load the RNN class and related functions
from eecs_545_project import PublisherClassifierStuff as pcs
# Load the best Vanilla RNN
BestVanillaRNN = pcs.PublisherClassifier(glove_dim=768, hidden_dim=128, cell_type='rnn')
BestVanillaRNN.load_state_dict(torch.load('nn_models/rnnBERT_dd768_hd128_1.pt'))
BestVanillaRNN.eval()

# Load the best LSTM model
BestLSTM = pcs.PublisherClassifier(glove_dim = 768, hidden_dim=128, cell_type='lstm')
BestLSTM.load_state_dict(torch.load('nn_models/lstmBERT_dd768_hd128_1.pt'))
BestLSTM.eval()

# Load the best LSTM-dropout model
BestDropoutLSTM = pcs.PublisherClassifier(glove_dim = 768, hidden_dim=128, cell_type='lstm_dropout')
BestDropoutLSTM.load_state_dict(torch.load('nn_models/lstmBERT_dropout_dd768_hd128_3.pt'))
BestDropoutLSTM.eval()



# Get glove vectors for the final testing dataset
glove_test_data, torch_data_labels = pcs.get_glove_data('eecs_545_project/test_dataset_final.csv', max_length=100)

data_labels = torch_data_labels.data.cpu().numpy()
print(data_labels)

# Get the accuracies from the best NNs
bestvanilla_preds = BestVanillaRNN.sample(torch_test_data).data.cpu().numpy()
print(bestvanilla_preds[:100])

bestlstm_preds = BestLSTM.sample(torch_test_data).data.cpu().numpy()
print(bestlstm_preds[:100])

bestdropout_preds = BestDropoutLSTM.sample(torch_test_data).data.cpu().numpy()
print(bestdropout_preds[:100])

# Load best LSA models
from joblib import dump, load
lsa_2_title_2 = load('eecs_545_project/lsa_2_title_2.joblib')
lsa_3_title_2 = load('eecs_545_project/lsa_3_title_2.joblib')
vectorizer_2_title_2 = load('eecs_545_project/vectorizer_2_title_2.joblib')
vectorizer_3_title_2 = load('eecs_545_project/vectorizer_3_title_2.joblib')
best_model_13 = load('eecs_545_project/best_model_13.joblib')
best_model_28 = load('eecs_545_project/best_model_28.joblib')
best_model_7 = load('eecs_545_project/best_model_7.joblib')

lsa_label_dict = {'Breitbart': 0, 'Washington Post': 1, 'CNN': 2, 'Business Insider': 3, 'Buzzfeed News': 4, 'NPR': 5, 'New York Times': 6, 'Reuters': 7, 'New York Post': 8, 'Vox': 9, 'Talking Points Memo': 10, 'Guardian': 11, 'Atlantic': 12, 'National Review': 13, 'Fox News': 14}
label_dict = {'New York Times': 0, 'Breitbart': 1, 'CNN': 2, 'Business Insider': 3, 'Atlantic': 4, 'Fox News': 5, 'Talking Points Memo': 6, 'Buzzfeed News': 7, 'National Review': 8, 'New York Post': 9, 'Guardian': 10, 'NPR': 11, 'Reuters': 12, 'Vox': 13, 'Washington Post': 14}
permute_dict = {0 : 1, 1 : 14, 2 : 2, 3 : 3, 4 : 7, 5 : 11, 6 : 0, 7 : 12, 8 : 9, 9 : 13, 10 : 6, 11 : 10, 12 : 4, 13 : 8, 14 : 5}

# Read test data into a pandas dataframe and run the LSA models on the test dataset

# read test_data_final.csv file into DataFrame and preprocess title
data_test_final = pd.read_csv('eecs_545_project/test_dataset_final.csv', sep=',', header=None).drop(labels=0, axis='columns')
data_test_final.columns = ['id', 'title', 'publication', 'author', 'date', 'year', 'month', 'url', 'content']

nlp = spacy.load('en', disable=['parser', 'ner'])

for j in range(len(data_test_final)):
    #Lemmatization
    doc = nlp(data_test_final.loc[j, 'title'])
    new_title = " ".join([token.lemma_ for token in doc])
    new_title = new_title.replace('-PRON- ', '')
    new_title = re.sub(r'[^a-zA-Z]', " ", new_title.lower())
    data_test_final.loc[j, 'title'] = new_title

# concatenate title
content_data_test_final = data_test_final[['title', 'content', 'publication']]
content_title_data_test_final = pd.DataFrame(columns=['content', 'publication'])
for j in range(len(content_data_test_final)):
    content_title_data_test_final.loc[j, 'publication'] = content_data_test_final.loc[j, 'publication']
    # article titles often have '- Publication' at the end of the title - remove this bc too big a hint
    corrected_title = re.sub((content_data_test_final.loc[j, 'publication']).lower(), '', content_data_test_final.loc[j, 'title'])
    content_title_data_test_final.loc[j, 'content'] = corrected_title + ' ' + content_data_test_final.loc[j, 'content']

# get final test sets
X_test_final = {'content' : content_data_test_final.loc[:, 'content'], 'title' : content_title_data_test_final.loc[:, 'content']}
y_test_final = {'content': [lsa_label_dict[pub] for pub in content_data_test_final.loc[:, 'publication']], 'title': [lsa_label_dict[pub] for pub in content_title_data_test_final.loc[:, 'publication']]}

'''
vectorizer_2_title_2 = joblib.load('vectorizer_2_title_2.joblib')
vectorizer_3_title_2 = joblib.load('vectorizer_3_title_2.joblib')
lsa_2_title_2 = joblib.load('lsa_2_title_2.joblib')
lsa_3_title_2 = joblib.load('lsa_3_title_2.joblib')
'''

X_test_tfidf_2_title_2 = vectorizer_2_title_2.transform(X_test_final['title'])
X_test_tfidf_3_title_2 = vectorizer_3_title_2.transform(X_test_final['title'])
X_test_lsa_2_title_2 = lsa_2_title_2.transform(X_test_tfidf_2_title_2)
X_test_lsa_3_title_2 = lsa_3_title_2.transform(X_test_tfidf_3_title_2)

# Get predictions from each of the lsa models
key = 'title'
lsa_28_preds = best_model_28.predict(np.array(X_test_lsa_3_title_2))

# Re-label the predictions for consistency with the other methods
for i in range(len(lsa_28_preds)):
  lsa_28_preds[i] = permute_dict[lsa_28_preds[i]]

lsa_7_preds = best_model_7.predict(np.array(X_test_lsa_2_title_2))
for i in range(len(lsa_7_preds)):
  lsa_7_preds[i] = permute_dict[lsa_7_preds[i]]

lsa_13_preds = best_model_13.predict(np.array(X_test_lsa_2_title_2))
for i in range(len(lsa_13_preds)):
  lsa_13_preds[i] = permute_dict[lsa_13_preds[i]]

def accuracy(preds, labels):
  num_correct = np.sum(np.equal(preds, labels))
  accuracy = num_correct.item() / len(labels)
  return accuracy

# A couple helper functions for showing confusion matrices

def confusion_mat_finder(preds, labels):
  confusion_mat = np.zeros((15, 15))
  for p, t in zip(preds, labels):
    confusion_mat[p, t] += 1
  return confusion_mat

def show_confusion_mat(confusion_mat, publisher_list): 
  plt.cla()
  plt.imshow(confusion_mat)
  
  plt.xlabel('Data labels')
  plt.ylabel('Predicted labels')
  plt.xticks(np.arange(15), publisher_list, rotation=90)
  plt.yticks(np.arange(15), publisher_list, rotation=0)
  plt.ylim(-0.5, 14.5)
  plt.gca().invert_yaxis()
  plt.colorbar()
  plt.show()

# Open the files with the QDA and LDA predictions
QDAfile = open('eecs_545_project/prediction_set_QDA.txt')
LDAfile = open('eecs_545_project/prediction_set_LDA.txt')
QDA_preds_str = QDAfile.read().split('\n')[:-1]
LDA_preds_str = LDAfile.read().split('\n')[:-1]

QDA_preds = []
for i in QDA_preds_str:
  if i != "[INVALID]":
    QDA_preds.append(label_dict[i])
  else:
    QDA_preds.append(-1)

LDA_preds = []
for i in LDA_preds_str:
  if i != "[INVALID]":
    LDA_preds.append(label_dict[i])
  else:
    LDA_preds.append(-1)

conf_mats = {}
accuracies = {}
accuracies['QDA'] = accuracy(QDA_preds, data_labels)
conf_mats['QDA'] = confusion_mat_finder(QDA_preds, data_labels)
accuracies['LDA'] = accuracy(LDA_preds, data_labels)
conf_mats['LDA'] = confusion_mat_finder(LDA_preds, data_labels)
accuracies['SVM with 2-Grams'] = accuracy(lsa_13_preds, data_labels)
conf_mats['SVM with 2-Grams'] = confusion_mat_finder(lsa_13_preds, data_labels)
accuracies['Logistic Regression with 3-Grams'] = accuracy(lsa_28_preds, data_labels)
conf_mats['Logistic Regression with 3-Grams'] = confusion_mat_finder(lsa_28_preds, data_labels)
accuracies['Random Forest with 2-Grams'] = accuracy(lsa_7_preds, data_labels)
conf_mats['Random Forest with 2-Grams'] = confusion_mat_finder(lsa_7_preds, data_labels)
accuracies['Vanilla RNN'] = accuracy(bestvanilla_preds, data_labels)
conf_mats['Vanilla RNN'] = confusion_mat_finder(bestvanilla_preds, data_labels)
accuracies['LSTM (no dropout)'] = accuracy(bestlstm_preds, data_labels)
conf_mats['LSTM (no dropout)'] = confusion_mat_finder(bestlstm_preds, data_labels)
accuracies['2-Layer LSTM with dropout'] = accuracy(bestdropout_preds, data_labels)
conf_mats['2-Layer LSTM with dropout'] = confusion_mat_finder(bestdropout_preds, data_labels)

for key in accuracies:
  print(key, accuracies[key])

for key in conf_mats:
  print(np.sum(conf_mats[key]))

pub_list = []
for key in label_dict:
  pub_list.append(key)


fig, axs = plt.subplots(4, 2, figsize=(20, 20))
fig.subplots_adjust(hspace=1, wspace=-0.5)

keys = conf_mats.keys

i = 0
j = 0
for key in (accuracies):
    print(j, i)
    ax = axs[i][j]
    im = ax.imshow(conf_mats[key],  aspect='equal')
    ax.set_title(key)
    ax.set_xlabel('Data labels')
    ax.set_ylabel('Predicted labels')
    ax.set_xticks(np.arange(15))
    ax.set_xticklabels(pub_list, rotation=90)
    ax.set_yticks(np.arange(15))
    ax.set_yticklabels(pub_list, rotation=0)
    ax.set_ylim(-0.5, 14.5)
    ax.invert_yaxis()
    fig.colorbar(im, ax=ax)
    im.set_clim(0, 95)

    j = (j + 1) % 2
    if j == 0:
      i = (i + 1) % 4
    


plt.show()
fig.savefig('confusion_matrices.png', bbox_inches='tight')

show_confusion_mat(LDA_conf_mat, pub_list)