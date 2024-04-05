from bs4 import BeautifulSoup as bs
import urllib.request as url
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import sent_tokenize
import heapq
import csv
import sys
from nltk.stem import PorterStemmer
import os


def summarize(source):
    soup = bs(source, 'lxml')
    x = PorterStemmer()
    imp_words = {}
    file_path = os.path.join('src', 'terms.csv')
    with open(file_path, mode='r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader, None)
        for row in csv_reader:
            key, value = row
            imp_words[key] = float(value)
    text = ''
    for para in soup.find_all('p'):
        text += para.get_text()
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text_modified = text.lower()
    text_modified = re.sub(r'\W', ' ', text_modified)
    text_modified = re.sub(r'\d', ' ', text_modified)
    text_modified = re.sub(r'\s+', ' ', text_modified)
    sentences = sent_tokenize(text)
    words = word_tokenize(text_modified)
    stoplist = stopwords.words('english')
    w2c = {}
    for word in words:
        w = x.stem(word)
        if w not in w2c.keys():
            w2c[w] = 1
        else:
            w2c[w] += 1
        for imp_word in imp_words:
            if imp_word in w:
                w2c[w] += imp_words[imp_word]
    for key in w2c:
        w2c[key] = w2c[key] / max(w2c.values())
    sent2score = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            w = x.stem(word)
            if w in w2c.keys():
                if len(sentence.split(' ')) < 25:
                    if sentence not in sent2score.keys():
                        sent2score[sentence] = w2c[w]
                    else:
                        sent2score[sentence] += w2c[w]
    best_sent = heapq.nlargest(20, sent2score, key=sent2score.get)
    return best_sent
