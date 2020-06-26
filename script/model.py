import os, re, csv, pickle
import pandas as pd
from nltk.stem.snowball import ItalianStemmer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaMulticore
from nltk.corpus import stopwords
from spacy.lang.it import Italian
import logging
import matplotlib.pyplot as plt


# logging.basicConfig(filename='old/gensim.log',
                    # format="%(asctime)s - %(levelname)s - %(message)s",
                    # level=logging.INFO)


it_stopwords = stopwords.words('italian')
it_stopwords.append("quest")

def load_docs(year, dir="data/converted/"):
    docs = []
    for txt in os.listdir(dir):
        if int(txt[1:5]) == year:
            text = open(dir+txt,"r",encoding='utf-8').read()
            docs.append(text)
    return docs

i = 0

def preprocess(text, NUM_DOCS, num_preprocessed, stemming):
    global i
    if i == 0:
        i = num_preprocessed
    i += 1
    result = []
    stemmer = ItalianStemmer()
    if i % 20 == 0:
        print(f"\t{i} out of {NUM_DOCS+num_preprocessed} documents preprocessed")
    nlp = Italian()
    t0 = text.split("Lingua processuale")[0].split("Sentenza")[-1]
    t1 = "".join(t0)
    t1 =  re.sub(r"’|'|«|»|\d{1,4}\/\d{1,4}\/(cee|ce)|\d+|---\|*|^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$", " ", t1, flags=re.IGNORECASE)
    # print(t1)
    doc = nlp(t1)
    for token in doc:
        if token.text.lower() not in it_stopwords and not token.is_punct | token.is_space and len(token) > 3:
            assert token.lang_ == "it"
            if stemming:
                result.append(stemmer.stem(word=token.text))
            else:
                result.append(token.lemma_.lower())
            if "'" in result[-1] or "’" in result[-1]:
                raise Exception(f"Detected_ {token.lemma_}")
    return result

def create_dict(corpus, NUM_TOPICS=5, filter_n_most_freq=10):
    dictionary = Dictionary(corpus)
    dictionary.filter_extremes(no_below=round(0.1/NUM_TOPICS*dictionary.num_docs))
    dictionary.filter_n_most_frequent(10)
    return dictionary


def compute_coherence_values(lda_model, corpus, dictionary, processed_docs, k, a, b):
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
    
    return coherence_model_lda.get_coherence()

def run_model(model, corpus, NUM_TOPICS, dictionary, save_file, plot_convergence=False):
    print(f"\nRunning model...")
    
    lda_model = model(corpus, num_topics=NUM_TOPICS, id2word=dictionary, 
                        chunksize=3000, 
                        eval_every=10, 
                        eta='symmetric',
                        alpha=0.31,
                        passes=30, 
                        iterations=200)
    # lda_model.save("data/.models/"+save_file)
    if plot_convergence:
        p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
        matches = [p.findall(l) for l in open('old/gensim.log', encoding='utf-8')]
        matches = [m for m in matches if len(m) > 0]
        tuples = [t[0] for t in matches]
        perplexity = [float(t[1]) for t in tuples]
        liklihood = [float(t[0]) for t in tuples]
        iter = list(range(0,len(tuples)*10,10))
        plt.plot(iter,liklihood,c="blue")
        plt.ylabel("log liklihood")
        plt.xlabel("iteration")
        plt.title("Topic Model Convergence")
        plt.grid()
        plt.savefig(f"fig/convergence_liklihood_{NUM_TOPICS}_{save_file}.png")
        plt.close()
    for idx, topic in lda_model.print_topics(-1):
        print('\nTopic: {} \nWords: {}'.format(idx, topic))
    return lda_model


def classify(lda_model, corpus, i, stemming, num=-1):
    print(f"\nClassification document {'last' if num==-1 else num} with model {i} and stemming={stemming}")
    for index, score in sorted(lda_model[corpus[num]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

def print_topics(model, corpus, output=None):
    if not output == None:
        file = open("data/output/"+output+".csv","w", newline='',encoding='utf-8')
        fwriter = csv.writer(file, delimiter=",")
    for topic in model.show_topics(num_topics=model.num_topics, formatted=False, num_words=25):
        row = []
        print(f"\nTopic {topic[0]}")
        for word in topic[1]:
            row.append(word)
            print(f"\t{word[0]}")
        if not output == None:
            fwriter.writerow(row)
    top_topics = model.top_topics(corpus) #, num_words=20)
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / model.num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

def print_bow(corpus, dictionary, num=0):
    for i in range(len(corpus[num])):
        print(dictionary.id2token[corpus[num][i][0]], corpus[num][i][1])


def plot_difference_plotly(mdiff, title="", annotation=None):
    """Plot the difference between models.

    Uses plotly as the backend."""
    import plotly.graph_objs as go
    import plotly.offline as py
    annotation_html = None
    if annotation is not None:
        annotation_html = [
            [
                "+++ {}<br>--- {}".format(", ".join(int_tokens), ", ".join(diff_tokens))
                for (int_tokens, diff_tokens) in row
            ]
            for row in annotation
        ]
    data = go.Heatmap(z=mdiff, colorscale='RdBu', text=annotation_html)
    layout = go.Layout(width=950, height=950, title=title, xaxis=dict(title="topic"), yaxis=dict(title="topic"))
    py.iplot(dict(data=[data], layout=layout))


def plot_difference_matplotlib(mdiff, title="", annotation=None):
    """Helper function to plot difference between models.

    Uses matplotlib as the backend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(title, size=14)
    plt.colorbar(data)
    plt.savefig(f"fig/{title}.png")


def topics_difference(first_model, second_model, title):
    try:
        get_ipython()
        import plotly.offline as py
    except Exception:
        #
        # Fall back to matplotlib if we're not in a notebook, or if plotly is
        # unavailable for whatever reason.
        #
        plot_difference = plot_difference_matplotlib
    else:
        py.init_notebook_mode()
        plot_difference = plot_difference_plotly
    mdiff, annotation = first_model.diff(second_model, distance='jaccard', num_words=50)
    plot_difference(mdiff, title=title+" (Jaccard distance)", annotation=annotation)
