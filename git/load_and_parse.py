#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:59:34 2019

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