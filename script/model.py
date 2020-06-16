import os, re
from nltk.stem.snowball import ItalianStemmer
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from spacy.lang.it import Italian

it_stopwords = stopwords.words('italian')
it_stopwords.append("quest")

def load_docs(from_year, dir="data/converted/"):
    docs = []
    for txt in os.listdir(dir):
        if int(txt[1:5]) >= from_year:
            text = open(dir+txt,"r",encoding='utf-8').read()
            docs.append(text)
    return docs

i = 0

def preprocess(text, NUM_DOCS):
    global i
    i += 1
    result = []
    stemmer = ItalianStemmer()
    if i % 20 == 0:
        print(f"{i} out of {NUM_DOCS} documents preprocessed")
    nlp = Italian()
    t0 = text.split("Lingua processuale")[0].split("Sentenza")[1:]
    t1 = "".join(t0)
    t1 =  re.sub(r"’|'|«|»|\d{1,4}\/\d{1,4}\/(cee|ce)|\d+|---\|*|^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$", " ", t1, flags=re.IGNORECASE)
    # print(t1)
    doc = nlp(t1)
    for token in doc:
        if token.text.lower() not in it_stopwords and not token.is_punct | token.is_space and len(token) > 3:
            assert token.lang_ == "it"
            result.append(stemmer.stem(word=token.lemma_))
            if "'" in result[-1] or "’" in result[-1]:
                raise Exception(f"Detected_ {token.lemma_}")
    return result

def create_dict(corpus, NUM_TOPICS=5, filter_n_most_freq=10):
    dictionary = Dictionary(corpus)
    dictionary.filter_extremes(no_below=round(0.5/NUM_TOPICS*dictionary.num_docs))
    dictionary.filter_n_most_frequent(10)
    return dictionary


def run_model(model, corpus, NUM_TOPICS, dictionary, save_file):
    print("\nRunning model...")
    lda_model = model(corpus, num_topics=NUM_TOPICS, id2word=dictionary)
    lda_model.save("data/.models/"+save_file)

    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    return lda_model


def classify(lda_model, corpus, num=1):
    print(f"\nClassification document {num} with model: {str(lda_model)}")
    for index, score in sorted(lda_model[corpus[num]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

def print_topics(model):
    for topic in model.show_topics(formatted=False):
        print(topic[0])
        for word in topic[1]:
            print(f"\t{word[0]}")

def print_bow(corpus, dictionary, num=0):
    for i in range(len(corpus[num])):
        print(dictionary.id2token[corpus[num][i][0]], corpus[num][i][1])
