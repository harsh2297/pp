# -*- coding: utf-8 -*-
import pandas as pd
import textblob
import nltk
import timeit

data = pd.read_csv("/Users/harsh/Desktop/text/merged.csv", header  = None)
data.columns = ['id','category','text']

print data[:5]
data['text'] = data['text'].apply(lambda x:" ".join(x.lower() for x in x.split()))
data['text'] = data['text'].str.replace('[^\w\s]', '')
print data[:5]
stop = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split()if x not in stop))
print data[:5]
from textblob import TextBlob
#data['headline'] = TextBlob(data['headline']).words
from textblob import Word
data['text'] = data['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))



data['text_sentiment'] = data['text'].apply(lambda x: TextBlob(str(x)).sentiment[0])
#data.to_csv("text_sentiment.csv")
