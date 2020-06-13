import os, pickle
import pandas as pd
from nltk.stem.snowball import SnowballStemmer, ItalianStemmer
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.test.utils import datapath
from nltk.corpus import wordnet, stopwords
import spacy
it_stopwords = stopwords.words('italian')


stemmer = SnowballStemmer('italian')
docs = []
for txt in os.listdir("data/converted"):
    text = open("data/converted/"+txt,"r",encoding='utf-8').read()
    docs.append(text)
    
print("Documents Loaded")

documents = pd.DataFrame()
documents["text"] = docs

i = 0

def preprocess(text):
    global i 
    i += 1
    if i % 20 == 0:
        print(f"{i} documents preprocessed")
    result = []
    nlp = spacy.load('it')
    doc = nlp(text)
    for token in doc:
        if token.text.replace("’",'').lower() not in it_stopwords and not token.is_punct | token.is_space and len(token) > 3 and token.text != "---|---":
            result.append(stemmer.stem(token.lemma_))
    return result

if os.path.exists("data/all_docs_preprocessed.pickle"):
    with open("data/all_docs_preprocessed.pickle","rb") as file:
        processed_docs = pickle.load(file)
    print("Documents preprocessed loaded")
else:
    print("Preprocesssing documents...")
    processed_docs = documents["text"].map(preprocess)
    with open("data/all_docs_preprocessed.pickle","wb") as file:
        pickle.dump(processed_docs, file)
    print("Documents preprocessed and saved")

dictionary = corpora.Dictionary(processed_docs)
print("Dictionary created:", dictionary)

print("Running model 1...")
corpus_bow = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(corpus_bow)
corpus_tfidf = tfidf[corpus_bow]

lda_model_bow = models.LdaModel(corpus_bow, num_topics=5, id2word=dictionary)
lda_model_bow.save("data/.models/bow")
# lda_model_bow = models.LdaModel.load("data/.models/bow")

print("\nModel 1 BOW")
for idx, topic in lda_model_bow.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

print("\nRunning model 2...")
lda_model_tfidf = models.LdaModel(corpus_tfidf, num_topics=5, id2word=dictionary)
lda_model_tfidf.save("data/.models/tfidf")

print("\nModel 2 TFIDF")
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

print("\nClassification document 100 - model 1:")
for index, score in sorted(lda_model_bow[corpus_bow[100]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_bow.print_topic(index, 10)))

print("\nClassification document 100 - model 2:")
for index, score in sorted(lda_model_tfidf[corpus_tfidf[100]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))