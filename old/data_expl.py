import os, pickle
from nltk.tokenize import word_tokenize
from spacy.lang.it import Italian
from gensim.corpora import Dictionary

average = 0
i = 0
for doc in os.listdir("data/converted/"):
    if int(doc[1:5]) >= 2000:
        i += 1
        text = word_tokenize(open("data/converted/"+doc,"r",encoding='utf-8').read(), language='italian')
        average += len(text)

print(average/i, i)

av = 0
i = 0
for dic in os.listdir("data/.preprocessed"):
    if dic[0] == 's':
        year = pickle.load(open("data/.preprocessed/"+dic,"rb"))
        i += len(year)
        print(len(year))
        for doc in year:
            av += len(doc)

print(av/i, i)

nlp = Italian()

doc = nlp(open("data/converted/61999CJ0001.txt","r",encoding='utf-8').read())
tok = []
for t in doc:
    tok.append(t.lemma_)

dictionary = Dictionary(tok)