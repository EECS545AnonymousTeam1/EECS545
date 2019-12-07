#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""



import numpy as np
import csv
import string
import sys
import re
import spacy

csv.field_size_limit(sys.maxsize)



#from string import maketrans

articles = []

sources = {"New York Times", "Breitbart", "CNN", "Business Insider", "The Atlantic", "Fox News", "Talking Points Memo", "Buzzfeed News", "National Review", "New York Post", "The Guardian", "NPR", "Reuters", "Vox","The Washington Post"}


with open('all-the-news/articles1.csv', newline='') as csvfile:
   reader = csv.reader(csvfile, delimiter=',', quotechar='"')
   categories = csvfile.readline()   # skip the first line
   for i, line in enumerate(reader):
       articles.append(line)
       
with open('all-the-news/articles2.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    categories = csvfile.readline()   # skip the first line
    for i, line in enumerate(reader):
        articles.append(line)

with open('all-the-news/articles3.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    categories = csvfile.readline()   # skip the first line
    for i, line in enumerate(reader):
        articles.append(line)


for i, article in enumerate(articles):
    if not article[9]:
        articles.remove(article)
    
    
shortened_articles = articles[0:260]
print(articles[0][3])
for i, article in enumerate(articles):
    if(i == len(articles) - 2):
        break
    if(articles[i][3] != articles[i+1][3]):
        shortened_articles.extend(articles[i+1:i+261])
        print(articles[i + 1][3])


nlp = spacy.load('en', disable=['parser', 'ner'])




# put all the words in the article together bc they get split by commas in the csvread
for i, shortened_article in enumerate(shortened_articles):
    #Lemmatization
    doc = nlp(shortened_article[9])
    shortened_article[9] = " ".join([token.lemma_ for token in doc])
    shortened_article[9] = shortened_article[9].replace('-PRON- ', '')
    #article[9] += article[9+j]
    #remove all punctuation from the articles
    shortened_article[9] = re.sub(r'[^a-zA-Z]', " ", shortened_article[9].lower())
        #print(shortened_article[9])
    shortened_articles[i] = shortened_article[0:10]



PGoutput = shortened_articles[0:10]
TRoutput = shortened_articles[10:210]
TEoutput = shortened_articles[210:260]
print(shortened_articles[1][3])
#gather playground data
for i in range(1,15):
    index = i * 260
    PGoutput.extend(shortened_articles[index:index+10])
    TRoutput.extend(shortened_articles[index+10:index+210])
    TEoutput.extend(shortened_articles[index+210:index+260])
    #print(shortened_articles[i+1][3])


with open('playground_dataset.csv', 'w', newline='') as csvfile:
    outwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for i in PGoutput:
        outwriter.writerow(i)
    
with open('train_dataset.csv', 'w', newline='') as csvfile:
    outwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for i in TRoutput:
        outwriter.writerow(i)
        
with open('test_dataset.csv', 'w', newline='') as csvfile:
    outwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    for i in TEoutput:
        outwriter.writerow(i)
        
        
#glove vector code
file1 = open("vectors.txt","r")
Dict = {} 
for line in file1:
    line = line.split()
    Dict[line[0]] = np.array((line[1:])).astype(float)
file1.close()


# In[3]:


# read in train and test csv's as pandas.DataFrame's
# pre-process the article titles in the same way as the article text

import pandas as pd

data_train = pd.read_csv('train_dataset.csv', sep=',', header=None).drop(labels=0, axis='columns')
#data_train.head(3)
data_train.columns = ['id', 'title', 'publication', 'author', 'date', 'year', 'month', 'url', 'content']
#data_train.head(3)

data_test = pd.read_csv('test_dataset.csv', sep=',', header=None).drop(labels=0, axis='columns')
data_test.columns = ['id', 'title', 'publication', 'author', 'date', 'year', 'month', 'url', 'content']
#data_test.head(3)

print(data_test['title'][0])
print(len(data_train))

# same pre-processing for title as article in case it is included in TF-IDF/LSA
for i in range(len(data_train)):
    #Lemmatization
    doc = nlp(data_train.loc[i, 'title'])
    new_title = " ".join([token.lemma_ for token in doc])
    new_title = new_title.replace('-PRON- ', '')
    new_title = re.sub(r'[^a-zA-Z]', " ", new_title.lower())
    data_train.loc[i, 'title'] = new_title
for i in range(len(data_test)):
    #Lemmatization
    doc = nlp(data_test.loc[i, 'title'])
    new_title = " ".join([token.lemma_ for token in doc])
    new_title = new_title.replace('-PRON- ', '')
    new_title = re.sub(r'[^a-zA-Z]', " ", new_title.lower())
    data_test.loc[i, 'title'] = new_title


# In[4]:


# get just the relevant data for our problem: title, content, publication
content_data_train = data_train[['title', 'content', 'publication']]
content_data_test = data_test[['title', 'content', 'publication']]
content_title_data_train = pd.DataFrame(columns=['content', 'publication'])
content_title_data_test = pd.DataFrame(columns=['content', 'publication'])

# merge content and title into one document
for i in range(len(content_data_train)):
    content_title_data_train.loc[i, 'publication'] = content_data_train.loc[i, 'publication']
    # article titles often have '- Publication' at the end of the title - remove this bc too big a hint
    # (slightly stupid right now because it leaves the 'the' in 'the new york times')
    corrected_title = re.sub((content_data_train.loc[i, 'publication']).lower(), '', content_data_train.loc[i, 'title'])
    content_title_data_train.loc[i, 'content'] = corrected_title + ' ' + content_data_train.loc[i, 'content']
for i in range(len(content_data_test)):
    content_title_data_test.loc[i, 'publication'] = content_data_test.loc[i, 'publication']
    # article titles often have '- Publication' at the end of the title - remove this bc too big a hint
    corrected_title = re.sub((content_data_test.loc[i, 'publication']).lower(), '', content_data_test.loc[i, 'title'])
    content_title_data_test.loc[i, 'content'] = corrected_title + ' ' + content_data_test.loc[i, 'content']
#content_data_train.head()


# In[5]:


# get train and test sets
X_train = {'content' : content_data_train.loc[:, 'content'], 'title' : content_title_data_train.loc[:, 'content']}
X_test = {'content' : content_data_test.loc[:, 'content'], 'title' : content_title_data_test.loc[:, 'content']}
unique_publications = list(set(content_data_train.loc[:, 'publication']))
label_dict = {}
for i in range(len(unique_publications)):
    label_dict[unique_publications[i]] = i
y_train = {'content': [label_dict[pub] for pub in content_data_train.loc[:, 'publication']], 'title': [label_dict[pub] for pub in content_title_data_train.loc[:, 'publication']]}
y_test = {'content': [label_dict[pub] for pub in content_data_test.loc[:, 'publication']], 'title': [label_dict[pub] for pub in content_title_data_test.loc[:, 'publication']]}


# In[30]:


# get tf-idf vectorizers (we want to tune max_df, so we try values in [0.25, 0.5, 1.])
# we also want to tune the largest n-grams considered, so 
from sklearn.feature_extraction.text import TfidfVectorizer

max_dfs = [0.25, 0.5, 1.]
max_ngram = [1, 2, 3]

vectorizers = {}
for max_n in max_ngram:
    vectorizers[max_n] = {'content': [], 'title': []}
    
    for max_df in max_dfs:
        vectorizers[max_n]['content'].append(TfidfVectorizer(max_df=max_df, min_df=10, stop_words='english', ngram_range=(1,max_n)))
        vectorizers[max_n]['title'].append(TfidfVectorizer(max_df=max_df, min_df=10, stop_words='english', ngram_range=(1,max_n)))
#print(len(vectorizers))

# get term-document matrices (entries are tf-idf for each word) for train and test set
X_train_tfidf = {}
for max_n in max_ngram:
    X_train_tfidf[max_n] = {}
    for k, v in X_train.items():
        X_train_tfidf[max_n][k] = []
        for i in range(len(max_dfs)):
            X_train_tfidf[max_n][k].append(vectorizers[max_n][k][i].fit_transform(X_train[k]))
            print('n_grams ', max_n, ' training ', k, ' max_df ', max_dfs[i], ' done')
        
X_test_tfidf = {}
for max_n in max_ngram:
    X_test_tfidf[max_n] = {}
    for k, v in X_test.items():
        X_test_tfidf[max_n][k] = []
        for i in range(len(max_dfs)):
            X_test_tfidf[max_n][k].append(vectorizers[max_n][k][i].transform(X_test[k]))
            print('n_grams ', max_n, ' test ', k, ' max_df ', max_dfs[i], ' done')


# In[31]:


from sklearn.decomposition import TruncatedSVD
import matplotlib
import matplotlib.pyplot as plt

# 1-grams, content, max_df=0.25
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[1]['content'][0]
print(len(X_test['content']))

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[1]['content'][0])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ncontent, max_df=%f, %d-grams' % (max_dfs[0], 1)))


# In[32]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

rank = 1000
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[1]['content'][0])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
lsa = {}
lsa[1] = {}
lsa[1]['content'] = []
(lsa[1]['content']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
X_train_lsa = {}
X_train_lsa[1] = {}
X_train_lsa[1]['content'] = []
(X_train_lsa[1]['content']).append(lsa[1]['content'][0].fit_transform(X_train_tfidf[1]['content'][0]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
X_test_lsa = {}
X_test_lsa[1] = {}
X_test_lsa[1]['content'] = []
(X_test_lsa[1]['content']).append(lsa[1]['content'][0].transform(X_test_tfidf[1]['content'][0]))


# In[33]:


# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[2]['content'][0]

# 2-grams, content, max_df=0.25
# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[2]['content'][0])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ncontent, max_df=%f, %d-grams' % (max_dfs[0], 2)))


# In[34]:


rank = 1050
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[2]['content'][0])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
lsa[2] = {}
lsa[2]['content'] = []
(lsa[2]['content']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
X_train_lsa[2] = {}
X_train_lsa[2]['content'] = []
(X_train_lsa[2]['content']).append(lsa[2]['content'][0].fit_transform(X_train_tfidf[2]['content'][0]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
X_test_lsa[2] = {}
X_test_lsa[2]['content'] = []
(X_test_lsa[2]['content']).append(lsa[2]['content'][0].transform(X_test_tfidf[2]['content'][0]))


# In[35]:


# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[3]['content'][0]

# 3-grams, content, max_df=0.25
# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[3]['content'][0])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ncontent, max_df=%f, %d-grams' % (max_dfs[0], 3)))


# In[36]:


rank = 1060
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[3]['content'][0])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
lsa[3] = {}
lsa[3]['content'] = []
(lsa[3]['content']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
X_train_lsa[3] = {}
X_train_lsa[3]['content'] = []
(X_train_lsa[3]['content']).append(lsa[3]['content'][0].fit_transform(X_train_tfidf[3]['content'][0]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
X_test_lsa[3] = {}
X_test_lsa[3]['content'] = []
(X_test_lsa[3]['content']).append(lsa[3]['content'][0].transform(X_test_tfidf[3]['content'][0]))


# In[37]:


# 1-grams, content, max_df=0.5
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[1]['content'][1]

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[1]['content'][1])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ncontent, max_df=%f, %d-grams' % (max_dfs[1], 1)))


# In[38]:


# 1000 also seems reasonable for this given scree plot
rank = 1000
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[1]['content'][1])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
(lsa[1]['content']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
(X_train_lsa[1]['content']).append(lsa[1]['content'][1].fit_transform(X_train_tfidf[1]['content'][1]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa[1]['content']).append(lsa[1]['content'][1].transform(X_test_tfidf[1]['content'][1]))


# In[39]:


# 2-grams, content, max_df=0.5
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[2]['content'][1]

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[2]['content'][1])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ncontent, max_df=%f, %d-grams' % (max_dfs[1], 2)))


# In[40]:


# 1000 also seems reasonable for this given scree plot
rank = 1050
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[2]['content'][1])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
(lsa[2]['content']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
(X_train_lsa[2]['content']).append(lsa[2]['content'][1].fit_transform(X_train_tfidf[2]['content'][1]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa[2]['content']).append(lsa[2]['content'][1].transform(X_test_tfidf[2]['content'][1]))


# In[41]:


# 3-grams, content, max_df=0.5
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[3]['content'][1]

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[3]['content'][1])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ncontent, max_df=%f, %d-grams' % (max_dfs[1], 3)))


# In[42]:


# 1000 also seems reasonable for this given scree plot
rank = 1060
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[3]['content'][1])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
(lsa[3]['content']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
(X_train_lsa[3]['content']).append(lsa[3]['content'][1].fit_transform(X_train_tfidf[3]['content'][1]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa[3]['content']).append(lsa[3]['content'][1].transform(X_test_tfidf[3]['content'][1]))


# In[43]:


# 1-grams, content, max_df=1.0
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[1]['content'][2]

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[1]['content'][2])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ncontent, max_df=%f, %d-grams' % (max_dfs[2], 1)))


# In[44]:


# 500 also seems reasonable for this given scree plot
rank = 1000
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[1]['content'][2])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
(lsa[1]['content']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
(X_train_lsa[1]['content']).append(lsa[1]['content'][2].fit_transform(X_train_tfidf[1]['content'][2]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa[1]['content']).append(lsa[1]['content'][2].transform(X_test_tfidf[1]['content'][2]))


# In[45]:


# 2-grams, content, max_df=1.0
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[2]['content'][2]

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[2]['content'][2])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ncontent, max_df=%f, %d-grams' % (max_dfs[2], 2)))


# In[46]:


# 500 also seems reasonable for this given scree plot
rank = 1050
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[2]['content'][2])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
(lsa[2]['content']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
(X_train_lsa[2]['content']).append(lsa[2]['content'][2].fit_transform(X_train_tfidf[2]['content'][2]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa[2]['content']).append(lsa[2]['content'][2].transform(X_test_tfidf[2]['content'][2]))


# In[47]:


# 3-grams, content, max_df=1.0
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[3]['content'][2]

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[3]['content'][2])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ncontent, max_df=%f, %d-grams' % (max_dfs[2], 3)))


# In[48]:


# 500 also seems reasonable for this given scree plot
rank = 1060
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[3]['content'][2])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
(lsa[3]['content']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
(X_train_lsa[3]['content']).append(lsa[3]['content'][2].fit_transform(X_train_tfidf[3]['content'][2]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa[3]['content']).append(lsa[3]['content'][2].transform(X_test_tfidf[3]['content'][2]))


# In[49]:


# now do same thing for pre-pending title of the article to content
# 1-grams, title, max_df=0.25
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[1]['title'][0]
print(len(X_test['title']))

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[1]['title'][0])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ntitle, max_df=%f, %d-grams' % (max_dfs[0], 1)))


# In[51]:


# 500 seems reasonable
rank = 1000
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[1]['title'][0])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
lsa[1]['title'] = []
(lsa[1]['title']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
X_train_lsa[1]['title'] = []
(X_train_lsa[1]['title']).append(lsa[1]['title'][0].fit_transform(X_train_tfidf[1]['title'][0]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
#X_test_lsa = {}
X_test_lsa[1]['title'] = []
(X_test_lsa[1]['title']).append(lsa[1]['title'][0].transform(X_test_tfidf[1]['title'][0]))


# In[52]:


# 2-grams, title, max_df=0.25
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[2]['title'][0]
print(len(X_test['title']))

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[2]['title'][0])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ntitle, max_df=%f, %d-grams' % (max_dfs[0], 2)))


# In[53]:


# 500 seems reasonable
rank = 1050
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[2]['title'][0])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
lsa[2]['title'] = []
(lsa[2]['title']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
X_train_lsa[2]['title'] = []
(X_train_lsa[2]['title']).append(lsa[2]['title'][0].fit_transform(X_train_tfidf[2]['title'][0]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
#X_test_lsa = {}
X_test_lsa[2]['title'] = []
(X_test_lsa[2]['title']).append(lsa[2]['title'][0].transform(X_test_tfidf[2]['title'][0]))


# In[54]:


# 3-grams, title, max_df=0.25
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[3]['title'][0]
print(len(X_test['title']))

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[3]['title'][0])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ntitle, max_df=%f, %d-grams' % (max_dfs[0], 3)))


# In[55]:


# 500 seems reasonable
rank = 1060
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[3]['title'][0])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
lsa[3]['title'] = []
(lsa[3]['title']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
X_train_lsa[3]['title'] = []
(X_train_lsa[3]['title']).append(lsa[3]['title'][0].fit_transform(X_train_tfidf[3]['title'][0]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
#X_test_lsa = {}
X_test_lsa[3]['title'] = []
(X_test_lsa[3]['title']).append(lsa[3]['title'][0].transform(X_test_tfidf[3]['title'][0]))


# In[56]:


# 1-grams, title, max_df=0.5
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[1]['title'][1]
print(len(X_test['title']))

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[1]['title'][1])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ntitle, max_df=%f, %d-grams' % (max_dfs[1], 1)))


# In[57]:


# 500 seems reasonable
rank = 1000
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[1]['title'][1])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
(lsa[1]['title']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
(X_train_lsa[1]['title']).append(lsa[1]['title'][1].fit_transform(X_train_tfidf[1]['title'][1]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa[1]['title']).append(lsa[1]['title'][1].transform(X_test_tfidf[1]['title'][1]))


# In[58]:


# 2-grams, title, max_df=0.5
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[2]['title'][1]
print(len(X_test['title']))

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[2]['title'][1])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ntitle, max_df=%f, %d-grams' % (max_dfs[1], 2)))


# In[59]:


# 500 seems reasonable
rank = 1050
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[2]['title'][1])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
(lsa[2]['title']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
(X_train_lsa[2]['title']).append(lsa[2]['title'][1].fit_transform(X_train_tfidf[2]['title'][1]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa[2]['title']).append(lsa[2]['title'][1].transform(X_test_tfidf[2]['title'][1]))


# In[60]:


# 3-grams, title, max_df=0.5
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[3]['title'][1]
print(len(X_test['title']))

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[3]['title'][1])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ntitle, max_df=%f, %d-grams' % (max_dfs[1], 3)))


# In[61]:


# 500 seems reasonable
rank = 1060
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[3]['title'][1])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
(lsa[3]['title']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
(X_train_lsa[3]['title']).append(lsa[3]['title'][1].fit_transform(X_train_tfidf[3]['title'][1]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa[3]['title']).append(lsa[3]['title'][1].transform(X_test_tfidf[3]['title'][1]))


# In[62]:


# 1-grams, title, max_df=1.0
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[1]['title'][2]
#print(len(X_test['title']))

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[1]['title'][2])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ntitle, max_df=%f, %d-grams' % (max_dfs[2], 1)))


# In[63]:


# 500 seems reasonable
rank = 1000
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[1]['title'][2])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
(lsa[1]['title']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
(X_train_lsa[1]['title']).append(lsa[1]['title'][2].fit_transform(X_train_tfidf[1]['title'][2]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa[1]['title']).append(lsa[1]['title'][2].transform(X_test_tfidf[1]['title'][2]))


# In[64]:


# 2-grams, title, max_df=1.0
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[2]['title'][2]
#print(len(X_test['title']))

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[2]['title'][2])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ntitle, max_df=%f, %d-grams' % (max_dfs[2], 2)))


# In[65]:


# 500 seems reasonable
rank = 1050
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[2]['title'][2])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
(lsa[2]['title']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
(X_train_lsa[2]['title']).append(lsa[2]['title'][2].fit_transform(X_train_tfidf[2]['title'][2]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa[2]['title']).append(lsa[2]['title'][2].transform(X_test_tfidf[2]['title'][2]))


# In[66]:


# 3-grams, title, max_df=1.0
# experiment with a good reduced rank to find one that explains most of the variance in the term-document matrices
X_train_tfidf[3]['title'][2]
#print(len(X_test['title']))

# plot the singular values (scree plot) to find a knee that suggests underlying low rank structure
rank = 2999
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[3]['title'][2])
print(svd.explained_variance_ratio_.sum())
fig, ax = plt.subplots()
ax.plot(svd.singular_values_)
ax.set(xlabel='n', ylabel='nth singular value', title = ('singular values of term-document matrix\ntitle, max_df=%f, %d-grams' % (max_dfs[2], 3)))


# In[67]:


# 500 seems reasonable
rank = 1060
svd = TruncatedSVD(rank)
svd.fit(X_train_tfidf[3]['title'][2])
print(svd.explained_variance_ratio_.sum())

# dictionary of TruncatedSVD transforms
(lsa[3]['title']).append(make_pipeline(svd, Normalizer(copy=False)))

# dictionary of tf-idf training data after TruncatedSVD transform
(X_train_lsa[3]['title']).append(lsa[3]['title'][2].fit_transform(X_train_tfidf[3]['title'][2]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa[3]['title']).append(lsa[3]['title'][2].transform(X_test_tfidf[3]['title'][2]))


# In[68]:


# sanity check lengths
print(len(X_train))
print(len(X_train['content']))
print(len(X_train['title']))
print(len(X_test))
print(len(X_test['content']))
print(len(X_test['title']))

print(len(X_train_tfidf))
for n in max_ngram:
    print(len(X_train_tfidf[n]['content']))
    print(X_train_tfidf[n]['content'][0].shape)
    print(X_train_tfidf[n]['content'][1].shape)
    print(X_train_tfidf[n]['content'][2].shape)
    print(len(X_train_tfidf[n]['title']))
    print(X_train_tfidf[n]['title'][0].shape)
    print(X_train_tfidf[n]['title'][1].shape)
    print(X_train_tfidf[n]['title'][2].shape)

print(len(X_test_tfidf))
for n in max_ngram:
    print(len(X_test_tfidf[n]['content']))
    print(X_test_tfidf[n]['content'][0].shape)
    print(X_test_tfidf[n]['content'][1].shape)
    print(X_test_tfidf[n]['content'][2].shape)
    print(len(X_test_tfidf[n]['title']))
    print(X_test_tfidf[n]['title'][0].shape)
    print(X_test_tfidf[n]['title'][1].shape)
    print(X_test_tfidf[n]['title'][2].shape)

print(len(lsa))
for n in max_ngram:
    print(len(lsa[n]['content']))
    print(len(lsa[n]['title']))

print(len(X_train_lsa))
for n in max_ngram:
    print(len(X_train_lsa[n]['content']))
    print(X_train_lsa[n]['content'][0].shape)
    print(X_train_lsa[n]['content'][1].shape)
    print(X_train_lsa[n]['content'][2].shape)
    print(len(X_train_lsa[n]['title']))
    print(X_train_lsa[n]['title'][0].shape)
    print(X_train_lsa[n]['title'][1].shape)
    print(X_train_lsa[n]['title'][2].shape)

print(len(X_test_lsa))
for n in max_ngram:
    print(len(X_test_lsa[n]['content']))
    print(X_test_lsa[n]['content'][0].shape)
    print(X_test_lsa[n]['content'][1].shape)
    print(X_test_lsa[n]['content'][2].shape)
    print(len(X_test_lsa[n]['title']))
    print(X_test_lsa[n]['title'][0].shape)
    print(X_test_lsa[n]['title'][1].shape)
    print(X_test_lsa[n]['title'][2].shape)

print(len(y_train))
print(len(y_train['content']))
print(len(y_train['title']))

print(len(y_test))
print(len(y_test['content']))
print(len(y_test['title']))


# In[109]:


from sklearn.model_selection import GridSearchCV, cross_val_score
import seaborn as sns

algorithm_comparison = pd.DataFrame(columns=['algorithm', 'n_train', 'n_test', 'use_title', 'max_df', 'n-gram', 'cv_score', 'train_acc', 'test_acc'])
cv_results = {'title': [None]*3, 'content': [None]*3}

label_dict = {'Breitbart': 0, 'Washington Post': 1, 'CNN': 2, 'Business Insider': 3, 'Buzzfeed News': 4, 'NPR': 5, 'New York Times': 6, 'Reuters': 7, 'New York Post': 8, 'Vox': 9, 'Talking Points Memo': 10, 'Guardian': 11, 'Atlantic': 12, 'National Review': 13, 'Fox News': 14}
label_dict_inv = {}
for k, v  in label_dict.items():
    label_dict_inv[v] = k
print(label_dict)
print(label_dict_inv)

print(y_test['title'])
blah = [1, 2, 3]
blahblah = [1, 3, 2]
blah = [label_dict_inv[j] for j in blah]
blahblah = [label_dict_inv[j] for j in blahblah]
result_dataframe = pd.DataFrame(columns=['true labels', 'pred labels'])
result_dataframe.loc[:,'true labels'] = blah
result_dataframe.loc[:,'pred labels'] = blahblah
mat = pd.crosstab(result_dataframe['true labels'], result_dataframe['pred labels'])
sns.heatmap(mat, annot=True)
plt.show()

best_models = {} # maps i (row in the DataFrame table) to the best model
best_conf_mat = {} # maps i (row in the DataFrame table) to the confusion matrix 
                    # (a cross-tabulated pd.DataFrame) for the best model

def algorithm_score(classifier, params, use_title, vectorizer_idx, i, algo_name):
    # use_title: True or False - if pre-pending title to article content
    # vectorizer_idx: (n_gram, idx)
    #      n_gram - 1, 2, 3 - max words used in n-grams
    #      idx - 0, 1, 2 - idx corresponding to max document frequency (0 has max_df=0.25, 1 has max_df=0.5, 2 has max_df=1.)
    key = 'title' if use_title else 'content'
    n = vectorizer_idx[0]
    df = vectorizer_idx[1]
    
    algorithm_comparison.loc[i, 'use_title'] = key
    algorithm_comparison.loc[i, 'max_df'] = max_dfs[df]
    algorithm_comparison.loc[i, 'n-gram'] = n
    
    print('use_title: ', key, ', max_df: ', max_dfs[df], 'n-gram: ', n)
    
    # 5-fold cross-validation
    grid_search = GridSearchCV(classifier, params, cv=5, verbose=2)
    grid_search.fit(np.array(X_train_lsa[n][key][df]), np.array(y_train[key]))
    cv_results[key][df] = grid_search.cv_results_
    
    # compute accuracies with best result from cross-validation
    best = grid_search.best_estimator_
    print('Best parameters: ', grid_search.best_params_)
    
    # show best scores for cross-validation
    cv_score = cross_val_score(best, X=np.array(X_train_lsa[n][key][df]), y = np.array(y_train[key]), cv=5)
    print('CV score for each fold (best estimator): ', cv_score)
    print('CV score averaged across folds (best estimator): ', cv_score.mean())
    algorithm_comparison.loc[i, 'cv_score'] = cv_score.mean()
    
    
    # use best parameters on whole training 
    best_all = best.fit(np.array(X_train_lsa[n][key][df]), np.array(y_train[key]))
    best_models[i] = best_all
    
    # use best parameters on whole training set for train accuracy
    train_acc = best_all.score(X=np.array(X_train_lsa[n][key][df]), y=np.array(y_train[key]))
    print('Training accuracy (best estimator): ', train_acc)
    algorithm_comparison.loc[i, 'train_acc'] = train_acc
    
    # use best parameters on whole training set for test accuracy
    test_acc = best_all.score(X=np.array(X_test_lsa[n][key][df]), y=np.array(y_test[key]))
    print('Test accuracy (best estimator): ', test_acc)
    algorithm_comparison.loc[i, 'test_acc'] = test_acc
    
    # create confusion matrix
    y_pred = best_all.predict(np.array(X_test_lsa[n][key][df]))
    #conf_mat = pd.crosstab(np.array(y_test[key]), y_pred)
    
    true_names = [label_dict_inv[j] for j in y_test[key]]
    pred_names = [label_dict_inv[j] for j in y_pred]
    result_dataframe = pd.DataFrame(columns=['true labels', 'predicted labels'])
    result_dataframe.loc[:,'true labels'] = true_names
    result_dataframe.loc[:,'predicted labels'] = pred_names

    plt.figure()
    conf_mat = pd.crosstab(result_dataframe['true labels'], result_dataframe['predicted labels'])
    best_conf_mat[i] = conf_mat
    sns.heatmap(conf_mat, annot=True)
    plt.title('Confusion Matrix: %s Classifier\n%s, max_df=%.2f, n-grams=%d' % (algo_name, key, max_dfs[df], n))
    plt.tight_layout()
    plt.savefig('confmat_%s_%s_%.2f_%d_i%d.png' % (algo_name, key, max_dfs[df], n, i))
    
    algorithm_comparison.loc[i, 'n_train'] = len(y_train[key])
    algorithm_comparison.loc[i, 'n_test'] = len(y_test[key])
    
    #return best_all
       


# In[92]:


from sklearn.ensemble import RandomForestClassifier

params = {'criterion': ['entropy', 'gini'],
         'n_estimators': [10, 50, 100]}

i = 0

for n in max_ngram:
    classifier = RandomForestClassifier(max_features=None, n_jobs=-1)
    print('----Random Forest-----')
    algorithm_comparison.loc[i, 'algorithm'] = 'RandomForest'
    # chose one with highest variance explained by SVD
    algorithm_score(classifier, params, True, (n, 2), i, 'RandomForest')
    i += 1


# In[95]:


algorithm_comparison.head(3)
img = plt.imread('confmat_RandomForest_title_1.00_1.png')
plt.imshow(img)
plt.show()
print(i)


# In[108]:


# test plotting the decision trees from a RandomForest classifier - https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c
from sklearn.datasets import load_iris
iris = load_iris()

# Model (can also use single decision tree)
model = RandomForestClassifier(n_estimators=10)

# Train
model.fit(iris.data, iris.target)
# Extract single tree
estimator = model.estimators_[5]

from sklearn.tree import export_graphviz
# Export as dot file
print(iris.feature_names)
print(iris.target_names)

export_graphviz(estimator, out_file='tree.dot', 
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = False)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')

class_names = [label_dict_inv[cls] for cls in range(15)]
print(label_dict)
print(label_dict_inv)
print(class_names)


# In[110]:


params = {'criterion': ['entropy', 'gini']}
class_names = [label_dict_inv[cls] for cls in range(15)]

for n in max_ngram:
    classifier = RandomForestClassifier(max_features=None, n_jobs=-1, n_estimators=200)
    print('----Random Forest-----')
    algorithm_comparison.loc[i, 'algorithm'] = 'RandomForest'
    # chose one with highest variance explained by SVD
    algorithm_score(classifier, params, True, (n, 2), i, 'RandomForest')
    
    # show 3 of the decision trees used
    for samp in range(3):
        est_idx = np.random.randint(0, high=200)
        estimator = best_models[i].estimators_[est_idx]
        fname = 'RandomForest_i%d_tree%d_%d' % (i, est_idx, samp)
        
        export_graphviz(estimator, out_file=(fname+'.dot'), class_names=class_names,
                       filled=False, rounded=True, precision=2, proportion=True)
        call(['dot', '-Tpng', fname+'.dot', '-o', fname+'.png'])
    i += 1


# In[114]:


print(i)
from IPython.display import SVG
from graphviz import Source
from IPython.display import display

graph = Source(export_graphviz(estimator, out_file=None,
               class_names=class_names,
               filled = True))
display(SVG(graph.pipe(format='svg')))


# In[115]:


n = 2
classifier = RandomForestClassifier(max_features=None, n_jobs=-1, n_estimators=200)
print('----Random Forest-----')
algorithm_comparison.loc[i, 'algorithm'] = 'RandomForest'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n, 2), i, 'RandomForest')
    
# show 1 of the decision trees used
est_idx = np.random.randint(0, high=200)
estimator = best_models[i].estimators_[est_idx]
fname = 'RandomForest_i%d_tree%d_%d' % (i, est_idx, samp)
 
graph = Source(export_graphviz(estimator, out_file=None,
               class_names=class_names,
               filled = True))
display(SVG(graph.pipe(format='svg')))
i += 1


# In[121]:


algorithm_comparison.head()
print(best_models[4])


# In[122]:


n = 3
classifier = RandomForestClassifier(max_features=None, n_jobs=-1, n_estimators=200)
print('----Random Forest-----')
algorithm_comparison.loc[i, 'algorithm'] = 'RandomForest'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n, 2), i, 'RandomForest')
    
# show 1 of the decision trees used
est_idx = np.random.randint(0, high=200)
estimator = best_models[i].estimators_[est_idx]
fname = 'RandomForest_i%d_tree%d_%d' % (i, est_idx, samp)
 
graph = Source(export_graphviz(estimator, out_file=None,
               class_names=class_names,
               filled = True))
display(SVG(graph.pipe(format='svg')))
i += 1


# In[124]:


algorithm_comparison.head(6)
print(best_models[5])


# In[125]:


print(cv_results)
algorithm_comparison.head(6)


# In[126]:


# try other parameters - use sqrt(n) vs n samples with replacement per tree (max_features)
#                      - use min samples to split decision tree (2 vs .05*n_train)
n = 1
params = {'criterion': ['entropy', 'gini'], 'min_samples_split': [2, .05], 'max_features': ['auto', None]}
classifier = RandomForestClassifier(max_features=None, n_jobs=-1, n_estimators=200)
print('----Random Forest-----')
algorithm_comparison.loc[i, 'algorithm'] = 'RandomForest'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n,2), i, 'RandomForest')

# show 1 of the decision trees used
est_idx = np.random.randint(0, high=200)
estimator = best_models[i].estimators_[est_idx]
fname = 'RandomForest_i%d_tree%d_%d' % (i, est_idx, samp)
 
graph = Source(export_graphviz(estimator, out_file=None,
               class_names=class_names,
               filled = True))
display(SVG(graph.pipe(format='svg')))
i += 1


# In[179]:


cv_results_table = pd.DataFrame(cv_results['title'][2])
cv_results_table.head(10)


# In[128]:


# try other parameters - use sqrt(n) vs n samples with replacement per tree (max_features)
#                      - use min samples to split decision tree (2 vs .05*n_train)
n = 2
params = {'criterion': ['entropy', 'gini'], 'min_samples_split': [2, .05], 'max_features': ['auto', None]}
classifier = RandomForestClassifier(max_features=None, n_jobs=-1, n_estimators=200)
print('----Random Forest-----')
algorithm_comparison.loc[i, 'algorithm'] = 'RandomForest'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n,2), i, 'RandomForest')

# show 1 of the decision trees used
est_idx = np.random.randint(0, high=200)
estimator = best_models[i].estimators_[est_idx]
fname = 'RandomForest_i%d_tree%d_%d' % (i, est_idx, samp)
 
graph = Source(export_graphviz(estimator, out_file=None,
               class_names=class_names,
               filled = True))
display(SVG(graph.pipe(format='svg')))
i += 1

algorithm_comparison.head(6)
# In[129]:


algorithm_comparison.head(6)


# In[131]:


# try other parameters - use sqrt(n) vs n samples with replacement per tree (max_features)
#                      - use min samples to split decision tree (2 vs .05*n_train)
n = 3
params = {'criterion': ['entropy', 'gini'], 'min_samples_split': [2, .05], 'max_features': ['auto', None]}
classifier = RandomForestClassifier(max_features=None, n_jobs=-1, n_estimators=200)
print('----Random Forest-----')
algorithm_comparison.loc[i, 'algorithm'] = 'RandomForest'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n,2), i, 'RandomForest')

# show 1 of the decision trees used
est_idx = np.random.randint(0, high=200)
estimator = best_models[i].estimators_[est_idx]
fname = 'RandomForest_i%d_tree%d_%d' % (i, est_idx, samp)
 
graph = Source(export_graphviz(estimator, out_file=None,
               class_names=class_names,
               filled = True))
display(SVG(graph.pipe(format='svg')))
i += 1


# In[132]:


algorithm_comparison.head(10)


# In[133]:


#print(y_train['title'])
#print(np.max(y_train['title']))
#print(np.array(y_train['title']) == 1)
class_frequencies = [np.sum(np.array(y_train['title']) == i) for i in range(np.max(y_train['title']))]
print('class frequencies: ', class_frequencies)
class_frequencies_test = [np.sum(np.array(y_test['title']) == i) for i in range(np.max(y_test['title']))]
print(class_frequencies_test)


# In[134]:


from sklearn.svm import SVC

n = 1
params = [{'kernel' : ['linear'], 'C': [1e-1, 1, 10]}, 
          {'kernel' : ['rbf'], 'C': [1e-1, 1, 10], 'gamma' : [1e-4, 1e-3, 1e-2, 1e-1]}]
classifier = SVC()
print('----Support Vector Machine-----')
algorithm_comparison.loc[i, 'algorithm'] = 'SVM'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n,2), i, 'SVM')
i += 1


# In[135]:


n = 2
params = [{'kernel' : ['linear'], 'C': [1e-1, 1, 10]}, 
          {'kernel' : ['rbf'], 'C': [1e-1, 1, 10], 'gamma' : [1e-4, 1e-3, 1e-2, 1e-1]}]
classifier = SVC()
print('----Support Vector Machine-----')
algorithm_comparison.loc[i, 'algorithm'] = 'SVM'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n,2), i, 'SVM')
i += 1


# In[137]:


algorithm_comparison.head(10)
print(best_models[9])


# In[138]:


n = 3
params = [{'kernel' : ['linear'], 'C': [1e-1, 1, 10]}, 
          {'kernel' : ['rbf'], 'C': [1e-1, 1, 10], 'gamma' : [1e-4, 1e-3, 1e-2, 1e-1]}]
classifier = SVC()
print('----Support Vector Machine-----')
algorithm_comparison.loc[i, 'algorithm'] = 'SVM'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n,2), i, 'SVM')
i += 1


# In[139]:


cv_results_table = pd.DataFrame(cv_results['title'][2])
cv_results_table.head(15)


# In[140]:


n = 1
params = [{'kernel' : ['linear'], 'C': [1, 2, 5, 10]}, 
          {'kernel' : ['rbf'], 'C': [5, 10, 20], 'gamma' : [5e-2, 1e-1, 1]}]
classifier = SVC()
print('----Support Vector Machine-----')
algorithm_comparison.loc[i, 'algorithm'] = 'SVM'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n,2), i, 'SVM')
i += 1


# In[141]:


n = 2
params = [{'kernel' : ['linear'], 'C': [1, 2, 5, 10]}, 
          {'kernel' : ['rbf'], 'C': [5, 10, 20], 'gamma' : [5e-2, 1e-1, 1]}]
classifier = SVC()
print('----Support Vector Machine-----')
algorithm_comparison.loc[i, 'algorithm'] = 'SVM'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n,2), i, 'SVM')
i += 1


# In[198]:


cv_results_table = pd.DataFrame(cv_results['title'][2])
cv_results_table.head(15)


# In[142]:


n = 3
params = [{'kernel' : ['linear'], 'C': [1, 2, 5, 10]}, 
          {'kernel' : ['rbf'], 'C': [5, 10, 20], 'gamma' : [5e-2, 1e-1, 1]}]
classifier = SVC()
print('----Support Vector Machine-----')
algorithm_comparison.loc[i, 'algorithm'] = 'SVM'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n,2), i, 'SVM')
i += 1


# In[199]:


params = {'kernel' : ['rbf'], 'C': [8, 10, 12, 15], 'gamma' : [8e-2, 1e-1, 2e-1, 5e-1]}
classifier = SVC()
print('----Support Vector Machine-----')
algorithm_comparison.loc[i, 'algorithm'] = 'SVM'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, 2, i)
i += 1


# In[201]:


cv_results_table = pd.DataFrame(cv_results['title'][2])
cv_results_table.head(16)


# In[143]:


from sklearn.linear_model import LogisticRegression

n = 1
params = {'solver' : ['newton-cg', 'sag'], 'C': [5e-1, 1, 10, 50]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n, 2), i, 'LogReg')
i += 1


# In[144]:


from sklearn.linear_model import LogisticRegression

n = 2
params = {'solver' : ['newton-cg', 'sag'], 'C': [5e-1, 1, 10, 50]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n, 2), i, 'LogReg')
i += 1


# In[145]:


n = 3
params = {'solver' : ['newton-cg', 'sag'], 'C': [5e-1, 1, 10, 50]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n, 2), i, 'LogReg')
i += 1


# In[146]:


n = 1
params = {'solver' : ['newton-cg', 'sag'], 'C': [5, 10, 15, 25]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n,2), i, 'LogReg')
i += 1


# In[147]:


n = 2
params = {'solver' : ['newton-cg', 'sag'], 'C': [5, 10, 15, 25]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n,2), i, 'LogReg')
i += 1


# In[148]:


n = 3
params = {'solver' : ['newton-cg', 'sag'], 'C': [5, 10, 15, 25]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n,2), i, 'LogReg')
i += 1


# In[149]:


n = 1
params = {'solver' : ['newton-cg', 'sag'], 'C': [3, 4, 5, 6, 7, 8, 9]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n, 2), i, 'LogReg')
i += 1


# In[150]:


n = 2
params = {'solver' : ['newton-cg', 'sag'], 'C': [3, 4, 5, 6, 7, 8, 9]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n, 2), i, 'LogReg')
i += 1


# In[151]:


n = 3
params = {'solver' : ['newton-cg', 'sag'], 'C': [3, 4, 5, 6, 7, 8, 9]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n, 2), i, 'LogReg')
i += 1


# In[154]:


algorithm_comparison.head(25)


# In[156]:


n = 1
params = {'solver' : ['newton-cg', 'sag'], 'C': [2, 3, 4, 5, 6, 7]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n, 2), i, 'LogReg')
i += 1


# In[157]:


n = 2
params = {'solver' : ['newton-cg', 'sag'], 'C': [2, 3, 4, 5, 6, 7]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n, 2), i, 'LogReg')
i += 1


# In[158]:


n = 1
params = {'solver' : ['newton-cg', 'sag'], 'C': [2.0, 2.5, 3., 3.5, 4.]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n, 2), i, 'LogReg')
i += 1


# In[159]:


n = 2
params = {'solver' : ['newton-cg', 'sag'], 'C': [2.0, 2.5, 3., 3.5, 4.]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n, 2), i, 'LogReg')
i += 1


# In[160]:


n = 3
params = {'solver' : ['newton-cg', 'sag'], 'C': [6.0, 6.5, 7., 7.5, 8.]}
classifier = LogisticRegression(penalty='l2', multi_class='multinomial')
print('----Logistic Regression-----')
algorithm_comparison.loc[i, 'algorithm'] = 'LogReg'
# chose one with highest variance explained by SVD
algorithm_score(classifier, params, True, (n, 2), i, 'LogReg')
i += 1


# In[412]:


# read test_data_final.csv file into DataFrame and preprocess title
toast = pd.read_csv('test_dataset_final.csv', sep=',', header=None, index_col=False)
data_test_final = toast
data_test_final = data_test_final.drop(labels=0, axis='columns')
#data_test_final = data_test_final.drop(labels=0, axis='columns')
data_test_final.columns = ['id', 'title', 'publication', 'author', 'date', 'year', 'month', 'url', 'content']

#idx_list = []
#for j in range(15):
    #idx_list += [2000*j + k for k in range(210)]
    #idx_list += [2000*j + k for k in range(260, 2000)]
#print(idx_list)
#data_test_final = data_test_final.drop(idx_list)
#data_test_final = data_test_final.reset_index()

#print('Dataframes equal?', pd.DataFrame.equals(data_test, data_test_final))
print(data_test_final)
print(data_test)

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
y_test_final = {'content': [label_dict[pub] for pub in content_data_test_final.loc[:, 'publication']], 'title': [label_dict[pub] for pub in content_title_data_test_final.loc[:, 'publication']]}

# get term-document matrices (entries are tf-idf for each word) for final test set
X_test_tfidf_final = {}
for max_n in max_ngram:
    X_test_tfidf_final[max_n] = {}
    for k, v in X_test_final.items():
        X_test_tfidf_final[max_n][k] = []
        for i in range(len(max_dfs)):
            X_test_tfidf_final[max_n][k].append(vectorizers[max_n][k][i].transform(X_test_final[k]))
            #print('n_grams ', max_n, ' test ', k, ' max_df ', max_dfs[i], ' done')

# dictionary of tf-idf final test data after TruncatedSVD transfrom (transform was learned from training data)
X_test_lsa_final = {}
X_test_lsa_final[1] = {}
X_test_lsa_final[1]['content'] = []
(X_test_lsa_final[1]['content']).append(lsa[1]['content'][0].transform(X_test_tfidf_final[1]['content'][0]))

# dictionary of tf-idf final test data after TruncatedSVD transfrom (transform was learned from training data)
X_test_lsa_final[2] = {}
X_test_lsa_final[2]['content'] = []
(X_test_lsa_final[2]['content']).append(lsa[2]['content'][0].transform(X_test_tfidf_final[2]['content'][0]))

# dictionary of tf-idf final test data after TruncatedSVD transfrom (transform was learned from training data)
X_test_lsa_final[3] = {}
X_test_lsa_final[3]['content'] = []
(X_test_lsa_final[3]['content']).append(lsa[3]['content'][0].transform(X_test_tfidf_final[3]['content'][0]))

# dictionary of tf-idf final test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa_final[1]['content']).append(lsa[1]['content'][1].transform(X_test_tfidf_final[1]['content'][1]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa_final[2]['content']).append(lsa[2]['content'][1].transform(X_test_tfidf_final[2]['content'][1]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa_final[3]['content']).append(lsa[3]['content'][1].transform(X_test_tfidf_final[3]['content'][1]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa_final[1]['content']).append(lsa[1]['content'][2].transform(X_test_tfidf_final[1]['content'][2]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa_final[2]['content']).append(lsa[2]['content'][2].transform(X_test_tfidf_final[2]['content'][2]))

(X_test_lsa_final[3]['content']).append(lsa[3]['content'][2].transform(X_test_tfidf_final[3]['content'][2]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
#X_test_lsa = {}
X_test_lsa_final[1]['title'] = []
(X_test_lsa_final[1]['title']).append(lsa[1]['title'][0].transform(X_test_tfidf_final[1]['title'][0]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
#X_test_lsa = {}
X_test_lsa_final[2]['title'] = []
(X_test_lsa_final[2]['title']).append(lsa[2]['title'][0].transform(X_test_tfidf_final[2]['title'][0]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
#X_test_lsa = {}
X_test_lsa_final[3]['title'] = []
(X_test_lsa_final[3]['title']).append(lsa[3]['title'][0].transform(X_test_tfidf_final[3]['title'][0]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa_final[1]['title']).append(lsa[1]['title'][1].transform(X_test_tfidf_final[1]['title'][1]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa_final[2]['title']).append(lsa[2]['title'][1].transform(X_test_tfidf_final[2]['title'][1]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa_final[3]['title']).append(lsa[3]['title'][1].transform(X_test_tfidf_final[3]['title'][1]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa_final[1]['title']).append(lsa[1]['title'][2].transform(X_test_tfidf_final[1]['title'][2]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa_final[2]['title']).append(lsa[2]['title'][2].transform(X_test_tfidf_final[2]['title'][2]))

# dictionary of tf-idf test data after TruncatedSVD transfrom (transform was learned from training data)
(X_test_lsa_final[3]['title']).append(lsa[3]['title'][2].transform(X_test_tfidf_final[3]['title'][2]))


# In[ ]:


vectorizer_2_title_2 = joblib.load('vectorizer_2_title_2.joblib')
vectorizer_3_title_2 = joblib.load('vectorizer_3_title_2.joblib')
lsa_2_title_2 = joblib.load('lsa_2_title_2.joblib')
lsa_3_title_2 = joblib.load('lsa_3_title_2.joblib')

X_test_tfidf_2_title_2 = vectorizer_2_title_2.transform(X_test_final['title'])
X_test_tfidf_3_title_2 = vectorizer_3_title_2.transform(X_test_final['title'])
X_test_lsa_2_title_2 = lsa_2_title_2.transform(X_test_tfidf_2_title_2)
X_test_lsa_3_title_2 = lsa_3_title_2.transform(X_test_tfidf_3_title_2)


# In[413]:


print(X_test_tfidf_final[1]['title'][2])
# sanity check lengths
print(len(X_test_final))
print(len(X_test_final['content']))
print(len(X_test_final['title']))

print(len(X_test_tfidf_final))
for n in max_ngram:
    print(len(X_test_tfidf_final[n]['content']))
    print(X_test_tfidf_final[n]['content'][0].shape)
    print(X_test_tfidf_final[n]['content'][1].shape)
    print(X_test_tfidf_final[n]['content'][2].shape)
    print(len(X_test_tfidf_final[n]['title']))
    print(X_test_tfidf_final[n]['title'][0].shape)
    print(X_test_tfidf_final[n]['title'][1].shape)
    print(X_test_tfidf_final[n]['title'][2].shape)

print(len(X_test_lsa_final))
for n in max_ngram:
    print(len(X_test_lsa_final[n]['content']))
    print(X_test_lsa_final[n]['content'][0].shape)
    print(X_test_lsa_final[n]['content'][1].shape)
    print(X_test_lsa_final[n]['content'][2].shape)
    print(len(X_test_lsa_final[n]['title']))
    print(X_test_lsa_final[n]['title'][0].shape)
    print(X_test_lsa_final[n]['title'][1].shape)
    print(X_test_lsa_final[n]['title'][2].shape)

print(len(y_test_final))
print(len(y_test_final['content']))
print(len(y_test_final['title']))

print('Dataframes equal?', pd.DataFrame.equals(data_test, data_test_final))
data_test_final.head()


# In[395]:


data_test.head()


# In[215]:


algorithm_comparison.head(40)


# In[216]:


print(best_models[7]) # best overall RandomForest (CV score)
print(best_models[13]) # best overall SVM (CV score)
print(best_models[28]) # best overall LogisticRegression (CV score)


# In[414]:


conf_mats_final = {}
print(X_test_final['title'][0])
print(label_dict_inv[y_test_final['title'][0]])
print(X_test['title'][0])
print(label_dict_inv[y_test['title'][0]])
for tbl_idx in [7, 13, 28]:
    print('--- ', algorithm_comparison.loc[tbl_idx, 'algorithm'], ', n-gram=', algorithm_comparison.loc[tbl_idx, 'n-gram'], ', title=', algorithm_comparison.loc[tbl_idx, 'use_title'], ', max_df=', algorithm_comparison.loc[tbl_idx, 'max_df'], ' ---')
    algo_name = algorithm_comparison.loc[tbl_idx, 'algorithm']
    n = algorithm_comparison.loc[tbl_idx, 'n-gram']
    key = algorithm_comparison.loc[tbl_idx, 'use_title']
    df = 2
    
    # use best parameters on whole training set for test accuracy
    test_acc = best_models[tbl_idx].score(X=np.array(X_test_lsa_final[n][key][df]), y=np.array(y_test_final[key]))
    print('Test accuracy (best estimator): ', test_acc)
    
    # create confusion matrix
    y_pred = best_models[tbl_idx].predict(np.array(X_test_lsa_final[n][key][df]))
    #conf_mat = pd.crosstab(np.array(y_test[key]), y_pred)
    
    true_names = [label_dict_inv[j] for j in y_test_final[key]]
    pred_names = [label_dict_inv[j] for j in y_pred]
    result_dataframe = pd.DataFrame(columns=['true labels', 'predicted labels'])
    result_dataframe.loc[:,'true labels'] = true_names
    result_dataframe.loc[:,'predicted labels'] = pred_names

    conf_mat = np.zeros((15, 15))
    plt.figure()
    for p, t in zip(y_pred, y_test_final[key]):
        conf_mat[p, t] += 1
    plt.imshow(conf_mat)
    #conf_mat = pd.crosstab(result_dataframe['true labels'], result_dataframe['predicted labels'])
    #best_conf_mat[i] = conf_mat
    #sns.heatmap(conf_mat, annot=True)
    #plt.title('Confusion Matrix: %s Classifier\n%s, max_df=%.2f, n-grams=%d' % (algo_name, key, max_dfs[df], n))
    #plt.tight_layout()
    #plt.savefig('final_confmat_%s_%s_%.2f_%d_i%d.png' % (algo_name, key, max_dfs[df], n, tbl_idx))


# In[ ]:


#best_model_7, 13, 28
# use best parameters on whole training set for test accuracy
test_acc = best_model_7.score(X=np.array(X_test_lsa_2_title_2), y=np.array(y_test_final[key]))
print('Test accuracy (best estimator): ', test_acc)


# In[232]:


plt.imshow(conf_mat)


# In[198]:


for n in [1, 2, 3]:
    for pc in [0, 1, 8, 9]:
        print('Principle Component ', pc, ':\n')
        neg_idxes = np.argsort(lsa[n]['title'][2].steps[0][1].components_[pc,:])
        pos_idxes = np.argsort(-np.array(lsa[n]['title'][2].steps[0][1].components_[pc,:]))
        
        print('10 most positive words:')
        for j in range(10):
            #print(n, pos_idxes[j])
            print(vectorizers[n]['title'][2].get_feature_names()[pos_idxes[j]], '\t', 
                 lsa[n]['title'][2].steps[0][1].components_[pc, pos_idxes[j]])
            
        print('\n10 most negative words:')
        for j in range(10):
            print(vectorizers[n]['title'][2].get_feature_names()[neg_idxes[j]], '\t',
                 lsa[n]['title'][2].steps[0][1].components_[pc, neg_idxes[j]])
        print('\n\n')
# TODO: put in nice format for the report        
        


# In[417]:


#2-GRAMS

# make scatter plots of (x0, x1), (x0, x8), (x1, x8)
# best classifier for these (retrain it on just the 2 points)
# plot decision boundary
# best models have n-grams: random forest: 2, SVM: 2, logisitic regression: 3
X_train_lsa_np = {}
X_train_lsa_np[2] = np.array(X_train_lsa[2]['title'][2])
X_train_lsa_np[3] = np.array(X_train_lsa[3]['title'][2])
y_train_np = np.array(y_train['title'])

legend_labels = [label_dict_inv[j] for j in range(15)]
    
print(X_train_lsa_np[2][:, 0].shape)
X0, X1, X8 = X_train_lsa_np[2][:, 0], X_train_lsa_np[2][:, 1], X_train_lsa_np[2][:, 8]
fig, ax = plt.subplots()
model_num = [7, 13, 28]
for m in model_num:
    print(best_models[m])
#print(X0.shape)
#print(X1.shape)
#print(X8.shape)
#print(y_train_np)
#print(X0[y_train_np==j].shape)
#print(X1[y_train_np==j].shape)
#print(X8[y_train_np==j].shape)
def get_meshgrid(x, y, delta=.02):
    x_min, x_max, = x.min()-.1, x.max()+.1
    y_min, y_max = y.min()-.1, y.max()+.1
    print('x', x_min, x_max)
    print('y', y_min, y_max)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, delta), np.arange(y_min, y_max, delta))
    return xx, yy
def plot_contours(ax, clf, xx, yy):
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    out = ax.contourf(xx, yy, z)
    return out

color = ['tab:blue', 'tab:orange']
cls_idx = [0, 6]

X_joined = np.concatenate((X0[:, np.newaxis], X1[:, np.newaxis]), axis=1)
X_joined = X_joined[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1]), :]
y_joined = y_train_np[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])]
model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=0.05,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=None, verbose=0,
                       warm_start=False)
clf = model.fit(X_joined, y_joined)
xx, yy = get_meshgrid(X0, X1)
plot_contours(ax, clf, xx, yy)

for j in range(2):
    ax.scatter(X0[y_train_np==cls_idx[j]], X1[y_train_np==cls_idx[j]], edgecolors='k', label=legend_labels[cls_idx[j]])
    ax.legend()
plt.title('Random Forest Decision Boundary:\nPublishers Along 0th and 1st Principal Component')
plt.xlabel('X0')
plt.ylabel('X1')


fig, ax = plt.subplots()

X_joined = np.concatenate((X0[:, np.newaxis], X8[:, np.newaxis]), axis=1)
X_joined = X_joined[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1]), :]
y_joined = y_train_np[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])]
model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=0.05,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=None, verbose=0,
                       warm_start=False)
clf = model.fit(X_joined, y_joined)
xx, yy = get_meshgrid(X0, X8)
plot_contours(ax, clf, xx, yy)

for j in range(2):
    ax.scatter(X0[y_train_np==cls_idx[j]], X8[y_train_np==cls_idx[j]], c=color[j], edgecolors='k', label=legend_labels[cls_idx[j]])
    ax.legend()
plt.title('Random Forest Decision Boundary:\nPublishers Along 0th and 8th Principal Component')
plt.xlabel('X0')
plt.ylabel('X8')

fig, ax = plt.subplots()

X_joined = np.concatenate((X1[:, np.newaxis], X8[:, np.newaxis]), axis=1)
X_joined = X_joined[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1]), :]
y_joined = y_train_np[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])]
model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=0.05,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=None, verbose=0,
                       warm_start=False)
clf = model.fit(X_joined, y_joined)
xx, yy = get_meshgrid(X1, X8)
plot_contours(ax, clf, xx, yy)

for j in range(2):
    ax.scatter(X1[y_train_np==cls_idx[j]], X8[y_train_np==cls_idx[j]], c=color[j], edgecolors='k', label=legend_labels[cls_idx[j]])
ax.legend()
plt.title('Random Forest Decision Boundary:\nPublishers Along 1st and 8th Principal Component')
plt.xlabel('X1')
plt.ylabel('X8')

# plot decision boundary after t-SNE

# make confusion matrices of shit data (X_test_final)


# In[418]:


# SVM decision boundaries on principle components (0, 1), (0, 8), (1, 8)

#0, 1
fig, ax = plt.subplots()
X_joined = np.concatenate((X0[:, np.newaxis], X1[:, np.newaxis]), axis=1)
X_joined = X_joined[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1]), :]
y_joined = y_train_np[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])]
model = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.05, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf = model.fit(X_joined, y_joined)
xx, yy = get_meshgrid(X0, X1)
plot_contours(ax, clf, xx, yy)

for j in range(2):
    ax.scatter(X0[y_train_np==cls_idx[j]], X1[y_train_np==cls_idx[j]], edgecolors='k', label=legend_labels[cls_idx[j]])
    ax.legend()
plt.title('SVM Decision Boundary:\nPublishers Along 0th and 1st Principal Component')
plt.xlabel('X0')
plt.ylabel('X1')

#0, 8
fig, ax = plt.subplots()
X_joined = np.concatenate((X0[:, np.newaxis], X8[:, np.newaxis]), axis=1)
X_joined = X_joined[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1]), :]
y_joined = y_train_np[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])]
model = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.05, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf = model.fit(X_joined, y_joined)
xx, yy = get_meshgrid(X0, X8)
plot_contours(ax, clf, xx, yy)

for j in range(2):
    ax.scatter(X0[y_train_np==cls_idx[j]], X8[y_train_np==cls_idx[j]], c=color[j], edgecolors='k', label=legend_labels[cls_idx[j]])
    ax.legend()
plt.title('SVM Decision Boundary:\nPublishers Along 0th and 8th Principal Component')
plt.xlabel('X0')
plt.ylabel('X8')


#1, 8
fig, ax = plt.subplots()
X_joined = np.concatenate((X1[:, np.newaxis], X8[:, np.newaxis]), axis=1)
X_joined = X_joined[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1]), :]
y_joined = y_train_np[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])]
model = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.05, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf = model.fit(X_joined, y_joined)
xx, yy = get_meshgrid(X1, X8)
plot_contours(ax, clf, xx, yy)

for j in range(2):
    ax.scatter(X1[y_train_np==cls_idx[j]], X8[y_train_np==cls_idx[j]], c=color[j], edgecolors='k', label=legend_labels[cls_idx[j]])
ax.legend()
plt.title('SVM Decision Boundary:\nPublishers Along 1st and 8th Principal Component')
plt.xlabel('X1')
plt.ylabel('X8')


# 

# In[419]:


# 3-GRAMS

# LogReg decision boundaries on principle components (0, 1), (0, 8), (1, 8)
X0, X1, X8 = X_train_lsa_np[3][:, 0], X_train_lsa_np[3][:, 1], X_train_lsa_np[3][:, 8]

#0, 1
fig, ax = plt.subplots()
X_joined = np.concatenate((X0[:, np.newaxis], X1[:, np.newaxis]), axis=1)
X_joined = X_joined[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1]), :]
y_joined = y_train_np[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])]
model = LogisticRegression(C=6.5, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='multinomial', n_jobs=None, penalty='l2',
                   random_state=None, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)
clf = model.fit(X_joined, y_joined)
xx, yy = get_meshgrid(X0, X1)
plot_contours(ax, clf, xx, yy)

for j in range(2):
    ax.scatter(X0[y_train_np==cls_idx[j]], X1[y_train_np==cls_idx[j]], edgecolors='k', label=legend_labels[cls_idx[j]])
    ax.legend()
plt.title('LogReg Decision Boundary:\nPublishers Along 0th and 1st Principal Component')
plt.xlabel('X0')
plt.ylabel('X1')

#0, 8
fig, ax = plt.subplots()
X_joined = np.concatenate((X0[:, np.newaxis], X8[:, np.newaxis]), axis=1)
X_joined = X_joined[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1]), :]
y_joined = y_train_np[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])]
model = LogisticRegression(C=6.5, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='multinomial', n_jobs=None, penalty='l2',
                   random_state=None, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)
clf = model.fit(X_joined, y_joined)
xx, yy = get_meshgrid(X0, X8)
plot_contours(ax, clf, xx, yy)

for j in range(2):
    ax.scatter(X0[y_train_np==cls_idx[j]], X8[y_train_np==cls_idx[j]], c=color[j], edgecolors='k', label=legend_labels[cls_idx[j]])
    ax.legend()
plt.title('LogReg Decision Boundary:\nPublishers Along 0th and 8th Principal Component')
plt.xlabel('X0')
plt.ylabel('X8')


#1, 8
fig, ax = plt.subplots()
X_joined = np.concatenate((X1[:, np.newaxis], X8[:, np.newaxis]), axis=1)
X_joined = X_joined[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1]), :]
y_joined = y_train_np[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])]
model = LogisticRegression(C=6.5, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='multinomial', n_jobs=None, penalty='l2',
                   random_state=None, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)
clf = model.fit(X_joined, y_joined)
xx, yy = get_meshgrid(X1, X8)
plot_contours(ax, clf, xx, yy)

for j in range(2):
    ax.scatter(X1[y_train_np==cls_idx[j]], X8[y_train_np==cls_idx[j]], c=color[j], edgecolors='k', label=legend_labels[cls_idx[j]])
ax.legend()
plt.title('LogReg Decision Boundary:\nPublishers Along 1st and 8th Principal Component')
plt.xlabel('X1')
plt.ylabel('X8')


# In[435]:


from sklearn.manifold import TSNE
from sklearn.neighbors.classification import KNeighborsClassifier

# 2-GRAM
#X_train_reduced_2 = TruncatedSVD(n_components=50).fit_transform(X_train_lsa_np[2])
X_train_embedded_2 = TSNE(n_components=2, perplexity=40., verbose=2).fit_transform((X_train_lsa_np[2])[:, :50])

x_tsne = X_train_embedded_2[:,0]
y_tsne = X_train_embedded_2[:,1]
#x_tsne = x_tsne[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])]
#y_tsne = y_tsne[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])]

# random forest best model
model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=0.05,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=None, verbose=0,
                       warm_start=False)
clf = model.fit((X_train_lsa_np[2])[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])],
                y_train_np[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])])               
y_train_embedded_2_pred = clf.predict(X_train_lsa_np[2])

# color points in grid as nearest neighbor of what their true point was classified as
fig, ax = plt.subplots()
one_nn = KNeighborsClassifier(n_neighbors=1).fit(X_train_embedded_2[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])],
                                                 y_train_embedded_2_pred[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])])
xx, yy = get_meshgrid(x_tsne, y_tsne, delta=1)
plot_contours(ax, one_nn, xx, yy)

# scatter plot of Breitbart and New York Times posts
for j in range(2):
    ax.scatter(x_tsne[y_train_np==cls_idx[j]], y_tsne[y_train_np==cls_idx[j]], edgecolors='k', label=legend_labels[cls_idx[j]])
    ax.legend()
plt.title('Approximate Random Forest Decision Boundary,\nFeatures Transformed with t-SNE')
#ax.scatter(X_train_embedded_2[:,0], X_train_embedded_2[:,1], c=y_train['title'], cmap=plt.cm.coolwarm, s=20, edgecolors='k')


# In[437]:


# SVM best model
model = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.05, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf = model.fit((X_train_lsa_np[2])[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])],
                y_train_np[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])])               
y_train_embedded_2_pred = clf.predict(X_train_lsa_np[2])

# color points in grid as nearest neighbor of what their true point was classified as
fig, ax = plt.subplots()
one_nn = KNeighborsClassifier(n_neighbors=1).fit(X_train_embedded_2[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])],
                                                 y_train_embedded_2_pred[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])])
xx, yy = get_meshgrid(x_tsne, y_tsne, delta=1)
plot_contours(ax, one_nn, xx, yy)

# scatter plot of Breitbart and New York Times posts
for j in range(2):
    ax.scatter(x_tsne[y_train_np==cls_idx[j]], y_tsne[y_train_np==cls_idx[j]], edgecolors='k', label=legend_labels[cls_idx[j]])
    ax.legend()
plt.title('Approximate SVM Decision Boundary,\nFeatures Transformed with t-SNE')


# In[438]:


# 2-GRAM
#X_train_reduced_2 = TruncatedSVD(n_components=50).fit_transform(X_train_lsa_np[2])
X_train_embedded_3 = TSNE(n_components=2, perplexity=40., verbose=2).fit_transform((X_train_lsa_np[3])[:, :50])

x_tsne = X_train_embedded_3[:,0]
y_tsne = X_train_embedded_3[:,1]
#x_tsne = x_tsne[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])]
#y_tsne = y_tsne[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])]

# random forest best model
model = LogisticRegression(C=6.5, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='multinomial', n_jobs=None, penalty='l2',
                   random_state=None, solver='newton-cg', tol=0.0001, verbose=0,
                   warm_start=False)
clf = model.fit((X_train_lsa_np[3])[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])],
                y_train_np[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])])               
y_train_embedded_3_pred = clf.predict(X_train_lsa_np[3])

# color points in grid as nearest neighbor of what their true point was classified as
fig, ax = plt.subplots()
one_nn = KNeighborsClassifier(n_neighbors=1).fit(X_train_embedded_3[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])],
                                                 y_train_embedded_3_pred[np.logical_or(y_train_np == cls_idx[0], y_train_np == cls_idx[1])])
xx, yy = get_meshgrid(x_tsne, y_tsne, delta=1)
plot_contours(ax, one_nn, xx, yy)

# scatter plot of Breitbart and New York Times posts
for j in range(2):
    ax.scatter(x_tsne[y_train_np==cls_idx[j]], y_tsne[y_train_np==cls_idx[j]], edgecolors='k', label=legend_labels[cls_idx[j]])
    ax.legend()
plt.title('Approximate Logistic Regression Decision Boundary,\nFeatures Transformed with t-SNE')
#ax.scatter(X_train_embedded_2[:,0], X_train_embedded_2[:,1], c=y_train['title'], cmap=plt.cm.coolwarm, s=20, edgecolors='k')


# In[254]:


#print(vectorizers[1]['title'][2].get_feature_names())
#print(vectorizers[2]['title'][2].get_feature_names())
#print(vectorizers[3]['title'][2].get_feature_names())
print(type(best_conf_mat[13]))
print(best_conf_mat[13])
print(best_conf_mat[13].to_numpy())

# TODO: convert to a numpy array so confusion matrix can be plotted for each top model
# TODO: confusion matrices for terrible results so that direct comparison can be made to RNN


# In[281]:


publisher_list = ['Atlantic', 'Breitbart', 'Business Insider', 'Buzzfeed News', 'CNN', 'Fox News', 'NPR', 'National Review', 'New York Post', 'New York Times', 'Reuters', 'Talking Points Memo', 'Vox', 'Washington Post']
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
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
    #plt.show()

for j in range(3, 29):
    plt.clf()
    show_confusion_mat(best_conf_mat[j].to_numpy())
    plt.title('Confusion Matrix: %s Classifier\n%s, max_df=%.2f, n-grams=%d' % (algorithm_comparison.loc[j, 'algorithm'], algorithm_comparison.loc[j, 'use_title'], algorithm_comparison.loc[j, 'max_df'], algorithm_comparison.loc[j, 'n-gram']), pad=10)
    plt.savefig('conf_mat_report_%d.png' % (j))


# In[270]:


print(label_dict)
for j in range(15):
    #print(X_test['title'][50*j])
    print(y_test_final['title'][2000*j])
    print(label_dict_inv[y_test_final['title'][2000*j]])


# In[253]:


# TODO: print most commonly occurring root nodes
root_nodes = [best_models[7].estimators_[j].tree_.feature[0] for j in range(200)]
unique, counts = np.unique(np.array(root_nodes), return_counts=True)
dic = dict(zip(unique, counts))
s_dic = sorted(dic.items(), key=lambda tup: tup[1])
print(s_dic[-5:]) # top 5 most common root nodes are 8, 17, 12, 1, 0; we expect 1 and 0 to be some of the best determining features because they explain most of the variance in the original dataset
# these are important features, so we will plot decsion boundary on X[0]-X[1] plane, and X[0]-X[8], X[1]-X[8] plane


# In[236]:


print(X_test_final['content'])


# In[408]:


from joblib import dump, load

dump(best_models[7], 'best_model_7.joblib')
dump(best_models[13], 'best_model_13.joblib')
dump(best_models[28], 'best_model_28.joblib')
dump(vectorizers[2]['title'][2], 'vectorizer_2_title_2.joblib')
dump(vectorizers[3]['title'][2], 'vectorizer_3_title_2.joblib')
dump(lsa[2]['title'][2], 'lsa_2_title_2.joblib')
dump(lsa[3]['title'][2], 'lsa_3_title_2.joblib')


# In[415]:


dump(best_models, 'best_model_dict.joblib')


# In[416]:


dump(lsa, 'lsa_dict.joblib')
dump(vectorizers, 'vectorizer_dict.joblib')


# In[ ]:




