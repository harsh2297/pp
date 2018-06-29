# -*- coding: utf-8 -*-
import pandas as pd
import textblob
import nltk
import timeit
data = pd.read_csv('/Users/harsh/Desktop/headline_data/only_headline.csv', header=None)
data.columns = ['headline']
data['word_count'] = data['headline'].apply(lambda x: len(str(x).split(" ")))
print(data[['headline', 'word_count']])
data.to_csv('headline_wordcount.csv')