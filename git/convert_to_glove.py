#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:44:34 2019

@author: nathanblock
"""
import numpy as np

#converts single word to glove vector
def convert_word_to_gloVe(word):
    return to_Glove[word] if word in to_Glove else np.array(None)

# def has_gloVe(word):
# 	return convert_word_to_gloVe(word).size != 1

#converts strings to list of glove vectors
def convert_article_to_gloVe(article):
    return [convert_word_to_gloVe(word) for word in article]
    
def create_gloVe_dict():
    file1 = open("vectors.txt","r")
    global to_Glove
    to_Glove = {}
    for line in file1:
        line = line.split()
        to_Glove[line[0]] = np.array((line[1:])).astype(float)
    file1.close()