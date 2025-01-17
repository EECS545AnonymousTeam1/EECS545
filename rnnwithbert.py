# -*- coding: utf-8 -*-
"""RNNWithBert.ipynb

Automatically generated by Colaboratory.

"""

# Commented out IPython magic to ensure Python compatibility.
# Starter code: setup, utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.nn.parameter import Parameter
import math
from getpass import getpass
import os

# Check whether GPU is available, and set our device to
# GPU if so.
# To enable GPU, Runtime -> Change Runtime Type -> Hardware Accelerator -> GPU.

if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')

print('Using device:', device)

# Copy the files from the eecs-545-project Gitlab repository
!rm -r eecs-545-project
!rm -r eecs_545_project

user = getpass('Gitlab user')
password = getpass('Gitlab password')
os.environ['GITLAB_AUTH'] = user + ':' + password


# Create a glove dictionary
from eecs_545_project import convert_to_glove_nn
convert_to_glove_nn.create_gloVe_dict('eecs_545_project/vectors.txt')

# Install BERT
!pip install pytorch-pretrained-bert

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
# % matplotlib inline

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# DO NOT RUN THIS CELL UNLESS YOU WANT TO MAKE MORE BERT VECTORS. Otherwise, load them from the files.
def convert_to_BERT(max_length=100, starting_ind = 0, num_articles_per_source=50, feature_type='word', num_sentences=10):

  '''
  Converts articles in the all-the-news CSV files to BERT vectors as inputs to the RNN.

  Inputs:
  - max_length: The maximum number of tokens to use from each article.
  - starting_ind: the first index within the list of articles to start collecting from (per source)
  - num_articles_per_source: the total number of articles per source to collect
  - feature_type: if 'word', creates a by-word representation of the features, if 'sentence', creates a sentence-avg version
  - num_sentences: the number of sentences per article to keep
  '''
  import csv
  import string
  import sys
  import re
  import spacy
  import time


  csv.field_size_limit(sys.maxsize)

  #from string import maketrans
  articles = []

  # Load pre-trained model (weights)
  model = BertModel.from_pretrained('bert-base-uncased')

  # Put the model in "evaluation" mode, meaning feed-forward operation.
  model.eval()

  with open('eecs_545_project/all-the-news/articles1.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    categories = csvfile.readline()   # skip the first line
    for i, line in enumerate(reader):
        articles.append(line)
        
  with open('eecs_545_project/all-the-news/articles2.csv', newline='') as csvfile:
      reader = csv.reader(csvfile, delimiter=',', quotechar='"')
      categories = csvfile.readline()   # skip the first line
      for i, line in enumerate(reader):
          articles.append(line)

  with open('eecs_545_project/all-the-news/articles3.csv', newline='') as csvfile:
      reader = csv.reader(csvfile, delimiter=',', quotechar='"')
      categories = csvfile.readline()   # skip the first line
      for i, line in enumerate(reader):
          articles.append(line)

  for i, article in enumerate(articles):
      if not article[9]:
          articles.remove(article)
  
  naps = int(num_articles_per_source)

  shortened_articles = articles[(starting_ind):(naps + starting_ind)]
  last_ind = 0
  print(articles[0][3])
  for i, article in enumerate(articles):
      if(i == len(articles) - 2):
          break
      if(articles[i][3] != articles[i+1][3]):
          shortened_articles.extend(articles[i+starting_ind+1:i+naps+starting_ind+1])
          print(articles[i + 1][3], i + 1 - last_ind)
          last_ind = i + 1

  feature_dim = 3072
  if feature_type == 'sentence':
    feature_dim = 768
  
  bert_length = max_length
  if feature_type == 'sentence':
    bert_length = num_sentences
  bert_articles = torch.zeros((15*naps, bert_length, feature_dim), dtype=torch.double, device='cuda:0', requires_grad=True)
  print(bert_articles.shape)
  
  nlp = spacy.load('en', disable=['parser', 'ner'])
  nlp.add_pipe(nlp.create_pipe('sentencizer')) # updated

  for i, article in enumerate(shortened_articles):
    start_t = time.time()
    print("Current article:", i, "publisher:", shortened_articles[i][3])
  
    # put all the words in the article together bc they get split by commas in the csvread
    article_text = ""
    for vec in article[9:]:
      article_text += vec

    # Remove casing in the article
    article_text = article_text.lower()
    #print("Article text:", shortened_articles[i][9])


    # Split the article into sentences
    doc = nlp(article_text)
    sentences = [sent.string.strip() for sent in doc.sents]
    sentences = [re.sub(r'[^a-zA-Z]', " ", sent) for sent in sentences]
    print("Number of sentences:", len(sentences))
    # Add sentence tokens to the input text
    sentence_iter = iter(sentences)
    sentence_pairs = zip(sentence_iter, sentence_iter)

    ind = 0
    sentence_ind = 0
    for pair in sentence_pairs:
      tokenized_sentence_1 = tokenizer.tokenize(pair[0])
      tokenized_sentence_2 = tokenizer.tokenize(pair[1])
      sent_1_len = len(tokenized_sentence_1) + 2
      tokenized_pair = ['[CLS]'] + tokenized_sentence_1 + ['[SEP]', '[CLS]'] + tokenized_sentence_2 + ['[CLS]']
      indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_pair)
      segment_ids = [0] * sent_1_len + [1] * (len(indexed_tokens) - sent_1_len)

      # Convert inputs to PyTorch tensors
      tokens_tensor = torch.tensor([indexed_tokens])
      segments_tensor = torch.tensor([segment_ids])
      
      # max_ind keeps the sentence pairs from being longer than 512 tokens (max size for BERT)
      max_ind = min(min(512, max_length - ind), len(tokenized_pair))
      #print(ind, max_ind)

      # Run through BERT model
      with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor[:, :max_ind], segments_tensor[:, :max_ind])

      # Concatenate the first four hidden layers of the network      
      if feature_type == 'word':
        bert_articles[i, ind:(ind + max_ind)] = torch.stack(encoded_layers, dim=0).squeeze(dim=1).permute(1, 0, 2)[:, -4:, :].reshape(max_ind, 3072)
      
      # Average the second-to-last layer of the network for each word in the sentence
      elif feature_type == 'sentence':
        #print("current sentence:", sentence_ind)
        bert_articles[i, sentence_ind] = torch.mean(torch.stack(encoded_layers, dim=0).squeeze(dim=1).permute(1, 0, 2)[:sent_1_len, -2, :], dim=0).reshape(1, 768)
        if sentence_ind + 1 >= num_sentences:
          break
        #print("current sentence:", sentence_ind + 1)
        bert_articles[i, sentence_ind + 1] = torch.mean(torch.stack(encoded_layers, dim=0).squeeze(dim=1).permute(1, 0, 2)[sent_1_len:, -2, :], dim=0).reshape(1, 768)
        sentence_ind += 2
      ind += max_ind
      
      if ind >= max_length or sentence_ind >= num_sentences:
        break
      end_t = time.time()

    print("Time for article:", round(end_t-start_t, 3))
      
  # Remove any nans
  print("Number of nans:", torch.sum(bert_articles != bert_articles))
  bert_articles[(bert_articles != bert_articles)] = 0.0
    
  return bert_articles

bert_articles = convert_to_BERT(max_length=1000, starting_ind=3000, num_articles_per_source=1, feature_type='sentence', num_sentences=10)
torch.save(bert_articles, 'eecs_545_project/test_test.pt')

# Install the PyDrive wrapper & import libraries.
# This only needs to be done once in a notebook.
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once in a notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive_2 = GoogleDrive(gauth)

# Create & upload a file.
uploaded = drive_2.CreateFile({'title': 'bert_articles2000_labels.pt'})
uploaded.SetContentFile('eecs_545_project/bert_articles2000_labels.pt')
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))

# Get precomputed BERT articles from Google Drive storage
from google.colab import drive
drive.mount('/content/drive')
bert_articles = torch.load('/content/drive/My Drive/bert_articles2000.pt')

# Get the labels for the BERT data
labels = torch.zeros(3000, dtype=torch.int64, device='cuda:0')
labels2 = torch.zeros(24000, dtype=torch.int64, device='cuda:0')
for i in range(15):
  labels[(200 * i):(200 * i + 200)] = i
  labels2[(1600 * i):(1600 * i + 1600)] = i
labels = torch.cat((labels, labels, labels2), dim=0)
print(labels.shape)
torch.save(labels, 'eecs_545_project/bert_articles2000_labels.pt')

rand_inds = torch.randperm(bert_articles.shape[0])

# The data is currently arranged by publisher. Randomize this for 
# easy splitting into test/validation sets.
bert_articles = bert_articles[rand_inds, :, :].clone().detach().requires_grad_(True)
labels = labels[rand_inds]

data = bert_articles.clone().detach().requires_grad_(True)

def get_article_data(filename, max_length):

  '''
  Inputs:
  - filename: a string with the name of the CSV file from which to read articles
  - max_length: the number of words to keep from each article

  Outputs:
  - glove_data: a torch tensor containing glove vectors for each word in each article;
              size: (num_articles, max_length, glove_length)
  - data_labels: a torch tensor of longs with labels from (0, 14) corresponding to the publishers
               of each article; size (num_articles)
  '''
  
  # Read in data from the playground dataset
  article_data = pd.read_csv(filename, header=None)

  # Place the article words into an array
  articles = article_data.values[:, 9]

  # Split the words in each article into a list
  split_articles = [article.split() for article in articles]

  num_articles = len(split_articles)
  gloVe_len = 50

  glove_data = np.zeros((num_articles, max_length, gloVe_len))

  for n in range(num_articles):
    glove_article = [convert_to_glove_nn.convert_word_to_gloVe(word) for word in split_articles[n]]
    glove_article = [i for i in glove_article if i.any()] # remove any None types
    len_art = len(glove_article)

    for j in range(min(max_length, len_art)):
      glove_data[n, j, :] = glove_article[j]
  
  glove_data = torch.tensor(glove_data, device='cuda:0', dtype=torch.double, requires_grad=True)

  publisher_list = "New York Times,Breitbart,CNN,Business Insider,Atlantic,Fox News,Talking Points Memo,Buzzfeed News,National Review,New York Post,Guardian,NPR,Reuters,Vox,Washington Post".split(',')

  data_publishers = article_data.values[:, 3]
  data_labels = np.zeros(len(data_publishers))

  for i in range(15):
    data_labels[(data_publishers == publisher_list[i])] = i
  
  data_labels = torch.tensor(data_labels, device='cuda:0', dtype=torch.int64)

  return glove_data, data_labels

test_glove, test_glove_labels = get_article_data('eecs_545_project/test_dataset_final.csv', max_length=100)

"""[1] Liuyu Zhou and Huafei Wang. News authorship identification with deep learning. Technical report, 2008.
URL https://cs224d.stanford.edu/reports/ZhouWang.pdf
"""

# Build a basic RNN, based on the architecture described in [1].
# At any point in time, outputs a confidence for a set of authors
# and a new hidden state.

def rnn_step_forward(x, prev_h, L, H, U, b1, b2):

  """
  Inputs:
  - x: Input data vector (GloVe vector for a word/sentence): (N, D)
  - prev_h: hidden state vector from previous timestep: (N, H)
  - L: affine transformation matrix for x: (D, H)
  - H: affine transformation matrix for prev_h: (H, H)
  - U: affine transformation matrix for y: (H, C)
  - b1: first bias vector: (H,)
  - b2: second bias vector: (C,)
  
  Returns:
  - next_h: new hidden state vector: (N, H)
  - y: prediction: (N, C)
  """
  e = x.mm(L) # e is shape (N, H)
  next_h = torch.sigmoid(prev_h.clone().mm(H) + e + b1.clone()) # new hidden state
  y = F.softmax(next_h.mm(U) + b2.clone(), dim=1) # New set of predictions for the authors

  return next_h, y

def rnn_forward(x, h0, L, H, U, b1, b2):

  """
  Inputs:
  - x: Input data vector (GloVe vector for a word/sentence): (N, T, D)
  - h0: initial hidden state vector: (N, H)
  - L: affine transformation matrix for x: (D, H)
  - H: affine transformation matrix for prev_h: (H, H)
  - U: affine transformation matrix for y: (H, C)
  - b1: first bias vector: (H,)
  - b2: second bias vector: (C,)
  
  Returns:
  - h: hidden states for entire timeseries: (N, T, H)
  - y: predictions at each timestep: (N, T, C)
  """
  
  # Get dimensions
  N, T, D = x.shape
  H_shape = h0.shape[1]
  C = b2.shape[0]

  # Initialize the hidden states and predictions arrays
  h = torch.zeros(N, T, H_shape, dtype = h0.dtype, device=h0.device)
  y = torch.zeros(N, T, C, dtype = h0.dtype, device=h0.device)
  next_h = h0

  for t in range(T):
    h[:, t, :], y[:, t, :] = rnn_step_forward(x[:, t, :], next_h, L, H, U, b1, b2)
    next_h = h[:, t, :]
  
  return h, y

# Define a class for the RNN
class VanillaRNN(nn.Module):
    
    def __init__(self, glove_dim=50, hidden_dim=128, num_classes=15, dtype=torch.double, device='cuda:0'):
        """
        
        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        super().__init__()
        
        # Initialize weights
        # Scaling weights are initialized randomly; biases are initialized to zero
        self.Wl = Parameter(torch.randn(glove_dim, hidden_dim, device=device, dtype=dtype, requires_grad=True).div(math.sqrt(glove_dim)))
        self.Wh = Parameter(torch.randn(hidden_dim, hidden_dim, device=device, dtype=dtype, requires_grad=True).div(math.sqrt(hidden_dim)))
        self.Wu = Parameter(torch.randn(hidden_dim, num_classes, device=device, dtype=dtype, requires_grad=True).div(math.sqrt(hidden_dim)))
        self.b1 = Parameter(torch.zeros(hidden_dim, device=device, dtype=dtype, requires_grad=True))
        self.b2 = Parameter(torch.zeros(num_classes, device=device, dtype=dtype, requires_grad=True))

        # Dimensions
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.glove_dim = glove_dim
    
    def forward(self, x, h0=None):
        """
        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - h0: Initial hidden state, of shape (N, H)

        Outputs:
        - hn: The hidden state output
        """
        N = x.shape[0]
        if h0 is None:
            h0 = torch.ones(N, self.hidden_dim, device=x.device, dtype=x.dtype, requires_grad=True)
        h, y = rnn_forward(x, h0, self.Wl, self.Wh, self.Wu, self.b1, self.b2)
        return y
  
    def step_forward(self, x, prev_h):
        """
        Inputs:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)

        Outputs:
        - next_h: The next hidden state, of shape (N, H)
        """
        next_h, _ = rnn_step_forward(x, prev_h, self.Wl, self.Wh, self.Wu, self.b1, self.b2)

        return next_h

# Modify the rnn's forward pass to use LSTM

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, Wu, b1, b2):

  """
  Inputs:
  - x: Input data vector (GloVe vector for a word/sentence): (N, D)
  - prev_h: hidden state vector from previous timestep: (N, H)
  - Wx: affine transformation matrix for x: (D, H)
  - Wh: affine transformation matrix for prev_h: (H, H)
  - U: affine transformation matrix for y: (H, C)
  - b1: first bias vector: (H,)
  - b2: second bias vector: (C,)
  
  Returns:
  - next_h: new hidden state vector: (N, H)
  - next_c: new cell state vector: (N, H)
  - y: prediction: (N, C)
  """

  # This code implements LSTM
  N, D = x.shape
  H = prev_h.shape[1]
  a = x.mm(Wx) + prev_h.mm(Wh) + b1

  i = torch.sigmoid(a[:, :H])
  f = torch.sigmoid(a[:, H:(2 * H)])
  o = torch.sigmoid(a[:, (2 * H):(3 * H)])
  tanh = nn.Tanh()
  g = tanh(a[:, (3 * H):].clone())

  next_c = f * prev_c + i * g
  next_h = o * tanh(next_c.clone())

  # Next prediction
  y = F.softmax(next_h.mm(Wu) + b2.clone(), dim=1) # New set of predictions for the authors

  return next_h, next_c, y

def lstm_forward(x, h0, Wx, Wh, Wu, b1, b2):

  """
  Inputs:
  - x: Input data vector (GloVe vector for a word/sentence): (N, T, D)
  - h0: initial hidden state vector: (N, H)
  - Wx: affine transformation matrix for x: (D, H)
  - Wh: affine transformation matrix for prev_h: (H, H)
  - Wu: affine transformation matrix for y: (H, C)
  - b1: first bias vector: (H,)
  - b2: second bias vector: (C,)
  
  Returns:
  - h: hidden state vectors for the entire timeseries: (N, T, H)
  - y: predictions: (N, T, C)
  """

  # Dimensions
  N, T, D = x.shape
  H = h0.shape[1]
  C = b2.shape[0]

  next_c = torch.zeros_like(h0)
  next_h = h0
  h = torch.zeros(N, T, H, device=h0.device, dtype=h0.dtype)
  y = torch.zeros(N, T, C, device=h0.device, dtype=h0.dtype)

  for t in range(T):
    next_h, next_c, y_t = lstm_step_forward(x[:, t, :], next_h, next_c, Wx, Wh, Wu, b1, b2)
    h[:, t, :] = next_h
    y[:, t, :] = y_t

  return h, y

# Define a class for the RNN
class LSTM(nn.Module):
    
    def __init__(self, glove_dim=50, hidden_dim=128, num_classes=15, dtype=torch.double, device='cuda:0'):
        """
        
        Inputs:
        - glove_dim: Dimension D of input gloVe feature vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        """
        super().__init__()
        
        # Initialize weights
        # Scaling weights are initialized randomly; biases are initialized to zero
        self.Wx = Parameter(torch.randn(glove_dim, hidden_dim * 4, device=device, dtype=dtype, requires_grad=True).div(math.sqrt(glove_dim)))
        self.Wh = Parameter(torch.randn(hidden_dim, hidden_dim * 4, device=device, dtype=dtype, requires_grad=True).div(math.sqrt(hidden_dim)))
        self.Wu = Parameter(torch.randn(hidden_dim, num_classes, device=device, dtype=dtype, requires_grad=True).div(math.sqrt(hidden_dim)))
        self.b1 = Parameter(torch.zeros(hidden_dim * 4, device=device, dtype=dtype, requires_grad=True))
        self.b2 = Parameter(torch.zeros(num_classes, device=device, dtype=dtype, requires_grad=True))

        # Dimensions
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.glove_dim = glove_dim
    
    def forward(self, x, h0=None):
        """
        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - h0: Initial hidden state, of shape (N, H)

        Outputs:
        - hn: The hidden state output
        """
        N = x.shape[0]
        if h0 is None:
            h0 = torch.ones(N, self.hidden_dim, device=x.device, dtype=x.dtype, requires_grad=True)
        # 
        h, y = lstm_forward(x, h0, self.Wx, self.Wh, self.Wu, self.b1, self.b2)
        return y
  
    def step_forward(self, x, prev_h, prev_c):
        """
        Inputs:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)

        Outputs:
        - next_h: The next hidden state, of shape (N, H)
        """
        next_h, _ = lstm_step_forward(x, prev_h, prev_c, self.Wx, self.Wh, self.Wu, self.b1, self.b2)

        return next_h

# Define a class for the 2-layer LSTM with dropout
class DropoutLSTM(nn.Module):
    
    def __init__(self, glove_dim=50, hidden_dim=128, num_classes=15, dtype=torch.double, device='cuda:0', dropout_prob=0.5):
        """
        
        Inputs:
        - glove_dim: Dimension D of input gloVe feature vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        """
        super().__init__()
        
        # Initialize weights
        # Scaling weights are initialized randomly; biases are initialized to zero
        #self.Wx1 = torch.empty(glove_dim, hidden_dim * 4)
        #torch.randn(size=(glove_dim, hidden_dim * 4), out=self.Wx1, dtype=dtype, layout=torch.strided, device=device, pin_memory=False, requires_grad=True)
        #self.Wx1 /= (math.sqrt(glove_dim))
        #self.Wx1 = Parameter(Wx1)
        self.Wx1 = Parameter(torch.randn(glove_dim, hidden_dim * 4, device=device, dtype=dtype, requires_grad=True).div(math.sqrt(glove_dim)))
        self.Wh1 = Parameter(torch.randn(hidden_dim, hidden_dim * 4, device=device, dtype=dtype, requires_grad=True).div(math.sqrt(hidden_dim)))
        self.Wx2 = Parameter(torch.randn(hidden_dim, hidden_dim * 4, device=device, dtype=dtype, requires_grad=True).div(math.sqrt(hidden_dim)))
        self.Wh2 = Parameter(torch.randn(hidden_dim, hidden_dim * 4, device=device, dtype=dtype, requires_grad=True).div(math.sqrt(hidden_dim)))
        self.Wu = Parameter(torch.randn(hidden_dim, num_classes, device=device, dtype=dtype, requires_grad=True).div(math.sqrt(hidden_dim)))
        self.b1 = Parameter(torch.zeros(hidden_dim * 4, device=device, dtype=dtype, requires_grad=True))
        self.b2 = Parameter(torch.zeros(hidden_dim * 4, device=device, dtype=dtype, requires_grad=True))
        self.b3 = Parameter(torch.zeros(num_classes, device=device, dtype=dtype, requires_grad=True))
      
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)

        # Dimensions
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.glove_dim = glove_dim
        
    
    def forward(self, x, h0=None):
        """
        Inputs:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - h0: Initial hidden state, of shape (2, N, H)

        Outputs:
        - hn: The hidden state output
        """
        N = x.shape[0]
        if h0 is None:
            h0 = torch.zeros(2, N, self.hidden_dim, device=x.device, dtype=x.dtype, requires_grad=True)
        h, y = self.forward_helper(x, h0)
        return y
  
    def step_forward(self, x, prev_h, prev_c):
        """
        Inputs:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)

        Outputs:
        - next_h: The next hidden state, of shape (N, H)
        """
        next_h, _ = self.step_forward_helper(x, prev_h, prev_c)

        return next_h

    def step_forward_helper(self, x, prev_h, prev_c):

      """
      Inputs:
      - x: Input data vector (GloVe vector for a word/sentence): (N, D)
      - prev_h: hidden state vector from previous timestep: (2, N, H) - 2 is for the 2 layers
      - prev_c: cell state from previous timestep: (2, N, H) - 2 is for the 2 layers
      - Wx1: affine transformation matrix on input for 1st layer: (D, 4H) - transforms input x to [i, f, o, g]
      - Wh1: affine transformation matrix on prev_h for 1st layer: (H, 4H) - transforms h_prev[0] to [i, f, o, g]
      - Wx2: affine transformation matrix on 1st layer output for 2nd layer: (H, 4H) - transforms input h_l1 to [i, f, o, g]
      - Wh2: affine transformation matrix on prev_h for 2nd layer: (H, 4H) - transforms h_prev[1] to [i, f, o, g]
      - Wu: affine transformation matrix for y: (H, C)
      - b1: first layer bias vector: (4H,)
      - b2: second layer bias vector: (4H,)
      - b3: output bias vector (for class prediction): (C,)
      - dropout: torch.nn.Dropout layer that will randomly drop some of the layer1-layer2 connections

      Returns:
      - next_h: new hidden state vector: (2, N, H) - 2 is for the 2 layers
      - next_c: new cell state vector: (2, N, H) - 2 is for the 2 layers
      - y: prediction: (N, C)
      """

      # This code implements LSTM
      N, D = x.shape
      H = prev_h.shape[2]
      next_c = torch.zeros_like(prev_c)
      next_h = torch.zeros_like(prev_h)

      # 1st layer (uses x as input)
      a1 = x.mm(self.Wx1) + prev_h[0,:,:].mm(self.Wh1) + self.b1

      i1 = torch.sigmoid(a1[:, :H])
      f1 = torch.sigmoid(a1[:, H:(2 * H)])
      o1 = torch.sigmoid(a1[:, (2 * H):(3 * H)])
      tanh = nn.Tanh()
      g1 = tanh(a1[:, (3 * H):].clone())

      next_c[0, :, :] = f1 * prev_c[0, :, :] + i1 * g1
      next_h[0, :, :] = o1 * tanh(next_c[0, :, :].clone())

      # 2nd layer (uses layer 1 output as input)
      a2 = (self.dropout(next_h[0,:,:])).mm(self.Wx2) + prev_h[1,:,:].mm(self.Wh2) + self.b2

      i2 = torch.sigmoid(a2[:, :H])
      f2 = torch.sigmoid(a2[:, H:(2 * H)])
      o2 = torch.sigmoid(a2[:, (2 * H):(3 * H)])
      g2 = tanh(a2[:, (3 * H):].clone())

      next_c[1,:,:] = f2 * prev_c[1, :, :] + i2 * g2
      next_h[1, :, :] = o2 * tanh(next_c[1, :, :].clone())

      # Next prediction
      y = F.softmax(next_h[1,:,:].mm(self.Wu) + self.b3.clone(), dim=1) # New set of predictions for the authors

      return next_h, next_c, y

    def forward_helper(self, x, h0):

      """
      Inputs:
      - x: Input data vector (GloVe vector for a word/sentence): (N, T, D)
      - h0: initial hidden state vector: (2, N, H) - 2 is for the 2 layers
      - Wx1: affine transformation matrix on input for 1st layer: (D, 4H) - transforms input x to [i, f, o, g]
      - Wh1: affine transformation matrix on prev_h for 1st layer: (H, 4H) - transforms h_prev[0] to [i, f, o, g]
      - Wx2: affine transformation matrix on 1st layer output for 2nd layer: (H, 4H) - transforms input h_l1 to [i, f, o, g]
      - Wh2: affine transformation matrix on prev_h for 2nd layer: (H, 4H) - transforms h_prev[1] to [i, f, o, g]
      - Wu: affine transformation matrix for y: (H, C)
      - b1: first layer bias vector: (4H,)
      - b2: second layer bias vector: (4H,)
      - b3: output class score bias vector: (C,)
  
      Returns:
      - h: hidden state vectors for the entire timeseries: (2, N, T, H)
      - y: predictions: (N, T, C)
      """

      # Dimensions
      N, T, D = x.shape
      H = h0.shape[2]
      C = self.num_classes

      next_c = torch.zeros_like(h0)
      next_h = h0
      h = torch.zeros(2, N, T, H, device=h0.device, dtype=h0.dtype)
      y = torch.zeros(N, T, C, device=h0.device, dtype=h0.dtype)

      for t in range(T):
        next_h, next_c, y_t = self.step_forward_helper(x[:, t, :], next_h, next_c)
        h[:, :, t, :] = next_h
        y[:, t, :] = y_t

      return h, y

# PublisherClassifier class
# This class will use an RNN to attempt to classify the publisher of articles.
class PublisherClassifier(nn.Module):

    def __init__(self, glove_dim=50, hidden_dim=128, num_classes=15,
                 cell_type='rnn', dtype=torch.double, device='cuda:0', dropout_prob=0.5):
        super().__init__()
        if cell_type not in {'rnn', 'lstm', 'lstm_dropout'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
  
        self.rnn = None
        self.lstm = None
        self.attn = None
        if self.cell_type == 'rnn':
          self.rnn = VanillaRNN(glove_dim=glove_dim, hidden_dim=hidden_dim,
                                num_classes=num_classes, dtype=dtype, device=device)
        
        elif self.cell_type == 'lstm':
          self.rnn = LSTM(glove_dim = glove_dim, hidden_dim = hidden_dim,
                          num_classes=num_classes, dtype=dtype, device=device)
        elif self.cell_type == 'lstm_dropout':
          self.rnn = DropoutLSTM(glove_dim=glove_dim, hidden_dim=hidden_dim, 
                                 num_classes=num_classes, dtype=dtype, device=device,
                                 dropout_prob=dropout_prob)

    
    def forward(self, article_data, labels):
      loss = 0.0
      scores = (self.rnn(article_data))[:, -1, :]

      loss = nn.functional.cross_entropy(scores, labels, reduction='mean')
    
      return loss

    def sample(self, glove_data):
      scores = (self.rnn(glove_data)[:, -1, :])
      preds = (torch.max(scores, axis=1))[1]
      return preds

import time
import copy

best_model_global = None
best_accuracy_global = -1.0

def accuracy(model, data, labels):
  preds = model.sample(data)
  num_correct = torch.sum(preds == labels)
  accuracy = num_correct.item() / len(labels)
  return accuracy

def NetworkTrainer(rnn_model, glove_data, data_labels, val_data = None,
                   val_labels = None, lr_decay=1, **kwargs):
  """
  Run optimization to train the model.

  Inputs:
  - rnn_model: The object we are training.
  - glove_data: the training feature data
  - label_data: the training labels
  - val_data: the validation feature data
  - val_labels: the validation labels
  """
  best_accuracy = best_accuracy_global
  best_model = best_model_global

  # optimizer setup
  from torch import optim
  optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, rnn_model.parameters()),
    learning_rate) # leave betas and eps by default
  lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                             lambda epoch: lr_decay ** epoch)

  # sample minibatch data
  iter_per_epoch = math.ceil(glove_data.shape[0] // batch_size)
  loss_history = []
  loss_history_smooth = []
  val_loss_history = []
  train_acc_history = []
  sf = 50 # smoothing factor
  val_acc_history = []

  rnn_model.train()
  for i in range(num_epochs):
    start_t = time.time()
    for j in range(iter_per_epoch):
      glove_vecs, labels = glove_data[j*batch_size:(j+1)*batch_size], \
                           data_labels[j*batch_size:(j+1)*batch_size]

      loss = rnn_model(glove_vecs, labels)
      optimizer.zero_grad()
      loss.backward()
      loss_history.append(loss.item())
      start_ind = max(0, len(loss_history) - sf)
      loss_history_smooth.append(np.mean(loss_history[start_ind:start_ind + sf]))

      optimizer.step()
    end_t = time.time()
    print('(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s'.format(
        i, num_epochs, loss.item(), end_t-start_t))
    train_acc_history.append(accuracy(rnn_model, glove_data, data_labels))
    print('Training accuracy:', train_acc_history[-1])
    if val_data is not None:
      val_acc_history.append(accuracy(rnn_model, val_data, val_labels))
      if val_acc_history[-1] > best_accuracy:
        best_accuracy = val_acc_history[-1]
        best_model = copy.deepcopy(rnn_model)
      print('Validation accuracy:', val_acc_history[-1])
      if best_model is not None:
        best_acc = accuracy(best_model, val_data, val_labels)
        print('Best accuracy: ', best_acc)
      

    lr_scheduler.step()

  # plot the training losses
  plt.plot(loss_history)
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title('Training loss history')

  plt.figure()
  plt.plot(loss_history_smooth)
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title('Training loss history (smoothed over ' + str(sf) + ' iterations)')

  plt.figure()
  plt.plot(train_acc_history)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title('Training accuracy history')

  if val_data is not None:
      plt.plot(val_acc_history)
      plt.legend(('training acc', 'validation acc'))

  plt.show()
  return best_model

# Try overfitting small data

model = PublisherClassifier(cell_type='lstm', glove_dim=768)

num_epochs = 50
batch_size = 10

for learning_rate in [5e-3]:
  print('learning rate is: ', learning_rate)
  best_model = NetworkTrainer(model, data[:100], labels[:100],
                num_epochs=num_epochs, batch_size=batch_size,
                learning_rate=learning_rate)

torch.save(model, 'test_model.pt')

train_data = data[:25000]
train_labels = labels[:25000]
val_data = data[25000:]
val_labels = labels[25000:]
#test_data = data[22000:]
#test_labels = labels[22000:]

print(val_data.shape)
print(val_labels.shape)
print(val_labels[0:100])

model2 = PublisherClassifier(glove_dim = 768, cell_type = 'lstm', hidden_dim=56)

num_epochs = 50
batch_size = 100

learning_rate = 1e-3
print('learning rate is: ', learning_rate)
best_model = NetworkTrainer(model2, train_data, train_labels,
                val_data=val_data, val_labels = val_labels,
                num_epochs=num_epochs, batch_size=batch_size,
                learning_rate=learning_rate, lr_decay=0.999)

test_data = torch.load('/content/drive/My Drive/bert_articles_for_testing.pt') # THIS WORKS
test_labels = torch.zeros(1500, dtype=torch.int64, device='cuda:0')
for i in range(15):
  test_labels[(100 * i):(100 * i + 100)] = i
print(test_labels)

model2.eval()
best_model.eval()
test_accuracy = accuracy(model2, test_data, test_labels)
print('Test accuracy:', test_accuracy)
train_accuracy = accuracy(model2, train_data, train_labels)
print('Train accuracy:', train_accuracy)
val_accuracy = accuracy(model2, val_data, val_labels)
print('Val_accuracy: ', val_accuracy)
best_model_accuracy = accuracy(best_model, test_data, test_labels)
print('Best model accuracy:', best_model_accuracy)

torch.save(best_model.state_dict(), 'rnnBERT_dd768_hd128_1.pt')

model3 = PublisherClassifier(cell_type='lstm', glove_dim=50, hidden_dim=40)
model3.load_state_dict(torch.load('first_good_lstm_model.pt'))
model3.eval()
test_accuracy = accuracy(model3, test_data_glove, test_glove_labels)
print(test_accuracy)

def accuracy_per_class(model, data, labels):
  preds = model.sample(data)
  num_articles = len(labels) / 15
  accuracies = []
  counts = []
  confusion_mat = torch.zeros((15, 15))

  for i in range(15):
    label_mask = (labels == i)
    num_correct = torch.sum(preds[label_mask] == labels[label_mask])
    class_accuracy = num_correct.item() / num_articles
    accuracies.append(class_accuracy)
    counts.append(torch.sum(preds == i).item())
  
  for p, t in zip(preds, labels):
    confusion_mat[p, t] += 1

  return accuracies, counts, confusion_mat

accuracies, counts, confusion_mat = accuracy_per_class(best_model, test_data, test_labels)
publisher_list = "New York Times,Breitbart,CNN,Business Insider,Atlantic,Fox News,Talking Points Memo,Buzzfeed News,National Review,New York Post,Guardian,NPR,Reuters,Vox,Washington Post".split(',')

for i in range(15):
  print("Publisher:", publisher_list[i], "- accuracy:", accuracies[i], "- total predictions:", counts[i])

def show_confusion_mat(confusion_mat): 
  plt.cla()
  plt.imshow(confusion_mat)
  
  plt.xlabel('Data labels')
  plt.ylabel('Predicted labels')
  
  #plt.xlim(-0.5, 14.5)
  #plt.ylim(-1, 15)
  plt.xticks(np.arange(15), publisher_list, rotation=90)
  plt.yticks(np.arange(15), publisher_list, rotation=0)
  plt.ylim(-0.5, 14.5)
  plt.gca().invert_yaxis()
  #plt.gca().set_aspect('equal', adjustable='box')
  plt.colorbar()
  plt.show()

accuracies, counts, confusion_mat = accuracy_per_class(model3, test_data_glove, test_glove_labels)
for i in range(15):
  print("Publisher:", publisher_list[i], "- accuracy:", accuracies[i], "- total predictions:", counts[i])

show_confusion_mat(confusion_mat)