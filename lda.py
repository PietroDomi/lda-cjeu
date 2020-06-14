import os, pickle, re
import pandas as pd
from nltk.stem.snowball import SnowballStemmer, ItalianStemmer
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.test.utils import datapath
from nltk.corpus import wordnet, stopwords
from spacy.lang.it import Italian,ItalianDefaults
from spacy_langdetect import LanguageDetector
it_stopwords = stopwords.words('italian')


stemmer = SnowballStemmer('italian')
docs = []
for txt in os.listdir("data/converted"):
    text = open("data/converted/"+txt,"r",encoding='utf-8').read()
    docs.append(text)

print("Documents Loaded")

documents = pd.DataFrame()
documents["text"] = docs

NUM_TOPICS = 5
i = 0

def preprocess(text):
    global i 
    i += 1
    if i % 20 == 0:
        print(f"{i} out of {len(docs)} documents preprocessed")
    result = []
    nlp = Italian()
    t0 = text.split("Lingua processuale")[0].split("Sentenza")[1:]
    t1 = "".join(t0)
    # print(t1)
    t1 =  re.sub(r"\d+|---\|*|^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$", "", t1, flags=re.IGNORECASE)
    doc = nlp(t1)
    for token in doc:
        if token.text.replace("’",'').lower() not in it_stopwords and not token.is_punct | token.is_space and len(token) > 3:
            assert token.lang_ == "it"
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
dictionary.filter_extremes(no_below=round(0.5/NUM_TOPICS*dictionary.num_docs))
dictionary.filter_n_most_frequent(10)
print("Dictionary created:", dictionary)

print(sorted(dictionary.cfs.values(),reverse=True))

print("\nRunning model 1...")
corpus_bow = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(corpus_bow)
corpus_tfidf = tfidf[corpus_bow]

lda_model_bow = models.LdaModel(corpus_bow, num_topics=NUM_TOPICS, id2word=dictionary)
lda_model_bow.save("data/.models/bow")
# lda_model_bow = models.LdaModel.load("data/.models/bow")

print("\nModel 1 BOW")
for idx, topic in lda_model_bow.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

print("\nRunning model 2...")
lda_model_tfidf = models.LdaModel(corpus_tfidf, num_topics=NUM_TOPICS, id2word=dictionary)
lda_model_tfidf.save("data/.models/tfidf")

print("\nModel 2 TFIDF")
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


def classify(num=1):
    print("\nClassification document {num} - model bow:")
    for index, score in sorted(lda_model_bow[corpus_bow[num]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model_bow.print_topic(index, 10)))

    print("\nClassification document {num} - model tfidf:")
    for index, score in sorted(lda_model_tfidf[corpus_tfidf[num]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

# classify(100)

for topic in lda_model_tfidf.show_topics(formatted=False):
    print(topic[0])
    for word in topic[1]:
        print(f"\t{word[0]}")


for i in range(len(corpus_bow[100])):
    print(dictionary.id2token[corpus_bow[100][i][0]],corpus_bow[100][i][1])