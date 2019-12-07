#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:41:22 2019

"""

import numpy as np
import csv
import string
import sys
import scipy
import itertools
import math
from scipy.spatial import distance_matrix
import re

from convert_to_glove import create_gloVe_dict
from convert_to_glove import convert_word_to_gloVe

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))



# Finds the source whose mean has minimum m_dist to word over all 
# the different sources whose means are given in means and whose 
# inverse covariance mat is given in siginv
def min_m_dist_lda(word, means, siginv):
    dist = np.zeros(len(means))
    for i, source in enumerate(means):
        mean = means[source]
        dist[i] = np.sqrt(np.matmul(np.matmul((word - mean).T, siginv), (word - mean)))
    return np.argmin(dist)

def min_m_dist_qda(word, means, siginv):
	# Finds the source whose mean has minimum m_dist to word over all 
	# the different sources whose means are given in means and whose 
	# inverse covariance mat is given in siginv
    # Differs from min_m_dist in that siginv here contains multiple matrices 
    # for all the different covariance matrices 
	dist = np.zeros(len(means))
	for i, source in enumerate(means):
		mean = means[source]
		dist[i] = np.sqrt(np.matmul(np.matmul((word - mean).T, siginv[source]), (word - mean)))
	return np.argmin(dist)

def min_m_dist(word, means, siginv, LDA):
    dist = np.zeros(len(means))
    for i, source in enumerate(means):
        mean = means[source]
        dist[i] = np.sqrt(np.matmul(np.matmul((word - mean).T, (siginv if LDA else siginv[source])), (word - mean)))
    return np.argmin(dist)


def populate_bag_of_words(corpus):
	# Populates bag of words with (occurrences, source, unique flag)
	# chosen_sources = ["Breitbart", "Buzzfeed News", "Reuters"]
	bag_of_words = {}
	for line in corpus:
		source = line[3]
		words = line[9]
		# if source in chosen_sources:
		for word in words.split(" "):
			if word not in bag_of_words:
				bag_of_words[word] = (0, source, True)
			flag = True
			if bag_of_words[word][2] == False or source != bag_of_words[word][1]:
				flag = False
			bag_of_words[word] = (bag_of_words[word][0]+1, source, flag)
	return bag_of_words

def populate_unique_bag_of_words(corpus):
	# Poulates bag of words with occurrence only in one source
	# and optionally, words that only appear more than k times

	# Populates bag of words with (occurences, source)
	mult_bag_of_words = populate_bag_of_words(corpus)
	appearance_threshold = 4

	bag_of_words = {}

	for word in mult_bag_of_words:
		if mult_bag_of_words[word][2] == True and mult_bag_of_words[word][0] > appearance_threshold:
			bag_of_words[word] = (mult_bag_of_words[word][0], mult_bag_of_words[word][1])

	return bag_of_words

def get_tfs(corpus):
	tfs = {} # maps words to dictionaries that map sources to tfs
	source_num_words = {}
	for line in corpus:
		source = line[3]
		words = line[9]
		split_words = words.split(" ")
		source_num_words[source] = len(split_words)
		for word in split_words:
			if convert_word_to_gloVe(word).size == 1:
				continue
			if word not in tfs:
				tfs[word] = {}
			if source not in tfs[word]:
				tfs[word][source] = 0
			tfs[word][source] += 1
	for word in tfs:
		for source in tfs[word]:
			tfs[word][source] /= source_num_words[source]
	return tfs

def get_word_ownership(tfs):
	ownership = {} # maps words to source
	for word in tfs:
		keys = list(tfs[word].keys())
		ownership[word] = keys[np.argmax(np.array([tfs[word][source] for source in tfs[word]]))]
	return ownership

def get_top_k_words(ownership, tf_idfs, k = 4000):
	source_words = {} # maps sources to list of (tf_idf, word)
	top_k_words = {} # maps sources to list of top k words rep. as (tf_idf, word)
	for word in ownership:
		source = ownership[word]
		if source not in source_words:
			source_words[source] = []
		source_words[source].append((tf_idfs[word][source], word))
	for source in source_words:
		top_k_words[source] = sorted(source_words[source], key = lambda x: x[0])
		top_k_words[source] = top_k_words[source][-k:]
	return top_k_words

def get_idf(corpus):
	counts = {} # maps words to number of documents the word appears in
	source_num_words = {}
	num_docs = 0
	for line in corpus:
		num_docs += 1
		words = line[9]
		split_words = words.split(" ")
		this_doc_words = set()
		for word in split_words:
			if convert_word_to_gloVe(word).size == 1:
				continue
			if word not in counts:
				counts[word] = 0
			if word not in this_doc_words:
				counts[word] += 1
				this_doc_words.add(word)
	idf_dict = {}
	for word in counts:
		idf_dict[word] = np.log(num_docs/counts[word])
    # print(idf_dict)
	return idf_dict


def get_tf_from_set(set_of_words, idf_dict):
	counts = {} # maps words to number of documents the word appears in
	source_num_words = {}
	num_docs = 0
	for word in set_of_words:
		if convert_word_to_gloVe(word).size == 1:
			continue
		if word not in counts:
			counts[word] = 0
		if word not in this_doc_words:
			counts[word] += 1
			this_doc_words.add(word)
	idf_dict = {}
	for word in counts:
		idf_dict[word] = np.log(num_docs/counts[word])
	# print(idf_dict)
	return idf_dict

def get_tf_idfs(tfs, idfs):
	tf_idfs = {}
	for word in tfs:
		for source in tfs[word]:
			if word not in tf_idfs:
				tf_idfs[word] = {}
			tf_idfs[word][source] = tfs[word][source]*idfs[word]
	return tf_idfs

def get_tf_idfs_single_article(tfs, idfs, num_docs):
	tf_idfs = {}
	for word in tfs:
		if word not in idfs:
			tf_idfs[word] = tfs[word]*np.log(num_docs+1)
		else:
			tf_idfs[word] = tfs[word]*idfs[word]
	return tf_idfs

def predict_article(words, idf_dict, means, covInv, num_training_docs, LDA, gamma = 50):
    source_list = list(means.keys())
    tfs = {}
    split_words = words.split(" ")
    for word in split_words:
        if convert_word_to_gloVe(word).size == 1:
            continue
        if word not in tfs:
            tfs[word] = 0
        tfs[word] += 1
    for word in tfs:
        tfs[word] /= len(split_words)
    tf_idfs = get_tf_idfs_single_article(tfs, idf_dict, num_training_docs)
    sorted_tf_idfs = sorted(tf_idfs.items(), key = lambda x : x[1])
    sorted_tf_idfs = sorted_tf_idfs[-gamma:]
    predictions = [ source_list[min_m_dist(convert_word_to_gloVe(word), means, covInv, LDA)] for word, tf_idf in sorted_tf_idfs ]
    
    
    
    
    return max(set(predictions), key=predictions.count)

def predict_corpus(corpus, idf_dict, means, covInv, num_training_docs, LDA, gamma=50):
	filename = 'prediction_set_LDA.txt' if LDA else 'prediction_set_QDA.txt'
	f = open(filename, 'w')
	num_correct = 0
	num_total = 0
	for line in corpus:
	    source = line[3]
	    words = line[9]
	    #some of the articles are empty bc they're photo galleries and such
	    if re.search('[a-zA-Z]', line[9]):
	        pred_source = predict_article(words, idf_dict, means, covInv, num_training_docs, LDA, gamma)
	        f.write(pred_source + '\n')
	        if pred_source == source:
	            num_correct += 1
	        num_total += 1
	    else:
	    	f.write('[INVALID]\n')
	percent_correct = num_correct / num_total
	f.close()
	return percent_correct

def get_corpus(file):
	with open(file, 'r', newline='') as csvfile:
		return list(csv.reader(csvfile, delimiter=',', quotechar='"'))

class LDA_QDA_Model:
	def __init__(self, LDA, k, gamma):
		self.LDA = LDA
		self.k = k
		self.gamma = gamma

	def fit(self, data):
		self.num_training_docs = len(data)
		self.tfs = get_tfs(data)
		self.idfs = get_idf(data)
		self.tf_idfs = get_tf_idfs(self.tfs, self.idfs)
		ownership = get_word_ownership(self.tfs)
		top_k_words = get_top_k_words(ownership, self.tf_idfs, self.k)

		using_bag_of_words = False

		bag_of_words = {}
		if using_bag_of_words:
			bag_of_words = populate_unique_bag_of_words(data)
		else:
			for source in top_k_words:
				for tup in top_k_words[source]:
					bag_of_words[tup[1]] = (0, source)
		source_words = {}
		# source_words_roman = {}

		# Populates dictionary source_words with keys as sources
		# and values as list of gloVe vector representations of
		# high frequency words unique to those sources
		for word in bag_of_words:
			if bag_of_words[word][1] not in source_words:
				source_words[bag_of_words[word][1]] = []
				# source_words_roman[bag_of_words[word][1]] = []
			source_words[bag_of_words[word][1]].append(convert_word_to_gloVe(word))
			# source_words_roman[bag_of_words[word][1]].append(word)

		# stacked_gloVes gives all the unique, high-freq words that
		# have gloVe vector representations
		# means stores the mean gloVe vector for each news source
		tot_size = 0
		stacked_gloVes = []
		self.means = {}
		covInv_qda = {}
		covariance = np.zeros([50, 50])
		for source in source_words:
			# size == 1 check corresponds to whether or not this word
			# had a valid gloVe representation. This is also used below
		    words = [word for word in source_words[source] if word.size != 1]
		    stacked_gloVes += words
		    tot_size += len(words)
		    self.means[source] = np.mean(words, axis=0)

		    if self.LDA:
		    	covariance += np.cov(np.array(words).T) # TODO bias = true
		    else:
			    # compute individual covariances for QDA. correction factor for very small eigenvalues that are going negative, 
			    # probably due to python bc Cov is PD ->inv(Cov) is PD too. So correct for it by computing eigendecomposition
			    # and setting negative values to positive
			    lam, evec = np.linalg.eig(np.cov(np.array(words).T))
			    
			    # correction for floating point messing with PD covariance structure
			    # cov_inv_qda = np.linalg.inv(np.cov(np.array(words).T))
			    # lam_inv, evec_inv = np.linalg.eig(cov_inv_qda)

			    # construct individual convariance for qda
			    covInv_qda[source] = np.real(np.matmul(evec.T, \
			    np.matmul(np.real(np.diag(abs(1./lam))), evec)))

		if self.LDA:
			# The covariance is the average of the covariance matrices
			# for each of the news sources; apparently taking the average
			# is a reasonable thing to do here since I am taking Nate's
			# notes at face value
			covariance /= tot_size
			self.covInv = np.linalg.inv(covariance)
		else:
			self.covInv = covInv_qda

	def predict(self, data):
		return predict_article(data, self.idfs, self.means, self.covInv, self.num_training_docs, self.LDA, self.gamma)

	def predict_corpus(self, data):
		return predict_corpus(data, self.idfs, self.means, self.covInv, self.num_training_docs, self.LDA, self.gamma)

def cross_validate(corpus, LDA, folds, k, gamma):
	fold_size = len(corpus)//folds
	total_acc = 0
	for i in range(folds):
		model = LDA_QDA_Model(LDA, k, gamma)
		model.fit(corpus[:i*fold_size] + corpus[(i+1)*fold_size:])
		# print("percent correct: " + str(model.predict_corpus(training)))

		total_acc += model.predict_corpus(corpus[i*fold_size:(i+1)*fold_size])
	total_acc /= folds
	return total_acc

csv.field_size_limit(sys.maxsize)

create_gloVe_dict()

train_data_file = 'train_dataset_final.csv'
train_data = get_corpus(train_data_file)

test_data_file = 'test_dataset_final.csv'
test_data = get_corpus(test_data_file)

# num_folds = 10
# err = []
# for k in range(100, 1000, 200):
# 	for gamma in range(10, 100, 20):
# 		print("k: " + str(k) + ", gamma: " + str(gamma) + " -- " + str(cross_validate(data, True, num_folds, k, gamma)))

# Best parameters found from cross validation were k = 300, gamma = 90
model = LDA_QDA_Model(True, 300, 90)
model.fit(train_data)
print('LDA Results:')
# print('Train Accuracy: ' + str(model.predict_corpus(train_data)))
print('Test Accuracy: ' + str(model.predict_corpus(test_data)))

model = LDA_QDA_Model(False, 300, 90)
model.fit(train_data)
print('QDA Results:')
# print('Train Accuracy: ' + str(model.predict_corpus(train_data)))
print('Test Accuracy: ' + str(model.predict_corpus(test_data)))

# We create a list of our sources, and calculate an accuracy
# score for our classifier
# source_list = list(means.keys())
# count = 0
# for word in bag_of_words:
#	if convert_word_to_gloVe(word).size == 1:
#		continue
#	if source_list[min_m_dist(convert_word_to_gloVe(word), means, covInv)] == bag_of_words[word][1]:
#		count += 1
# print(count/len(stacked_gloVes))








