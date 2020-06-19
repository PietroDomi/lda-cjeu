import pandas as pd
from script import model, explore, html_converter
import os, pickle, argparse, json
from gensim.models import TfidfModel, LdaModel


def main():

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="LDA")
    parser.add_argument("--year", type=int, metavar="FROM_YEAR", help="what year you would like to analyse the data from (year, corpus size): \n(2015, 7), (2014, 113), (2013, 132), (2012, 159), (2011, 242), (2010, 612), (2009, 1090), (2008, 1291), (2007, 1431), (2006, 1720), (2005, 2162), (2004, 2561), (2003, 2983), (2002, 3349), (2001, 3795)", default=2010)
    parser.add_argument("--k", type=int, metavar="NUM_TOPICS", help="number of topics for LDA", default=5)
    parser.add_argument("--stem", type=str2bool, metavar="STEMMING", help="whether to stem the words (default)", default=True)
    parser.add_argument("--plot", type=str2bool, metavar="SHOW DISTR PLOT", help="displays the distribution of documents through time", default=False)
    args = parser.parse_args()

    NUM_TOPICS = args.k
    SHOW_DISTR_PLOT = args.plot
    FROM_YEAR = args.year
    STEMMING = args.stem
    converted = False

    print("\nChecking for preprocessed documents...")

    to_load = []
    processed_docs = pd.Series(dtype=object)
    for year in range(2015, FROM_YEAR-1, -1):
        try:
            prepro_docs = pickle.load(open(f"data/.preprocessed/{'s' if STEMMING else 'l'}_{year}_preprocessed.pickle","rb"))
            processed_docs = pd.concat([processed_docs,prepro_docs], axis=0)
        except:
            to_load.append(year)
    print("\nDocuments Loaded")
    

    if os.path.exists("data/converted") and int(os.listdir("data/converted")[0][1:5]) > FROM_YEAR:
        print("\nConverting data from html to txt...")
        NUM_DOCS = html_converter.convert()
        converted = True

    elif os.path.exists("data_scraping/data_html") and int(os.listdir("data_scraping/data_html")[0][1:5]) > FROM_YEAR:
        print("\nCollecting data...")
        explore.get(from_year=FROM_YEAR, show=SHOW_DISTR_PLOT)
        os.system("cd data_scraping && scrapy crawl celex")
        print("\n\nData Collected")
        NUM_DOCS = html_converter.convert()
        converted = True
        print(f"\nData converted: {NUM_DOCS} total documents")
        with open(".gitignore","a") as file:
            celexs = open("data_scraping/to_get.txt","r").readlines()
            for celex in celexs:
                file.writelines("data_scraping/data_html/"+celex[:-1]+".html\n")
                file.writelines("data/converted/"+celex[:-1]+".txt\n")

    if not converted:
        NUM_DOCS = html_converter.convert()

    for year in sorted(to_load, reverse=True):
        print(f"\nProcesssing documents for {year}...")
        documents = pd.Series(model.load_docs(year, dir="data/converted/"))
        prepro_doc = documents.apply(func=model.preprocess, args=(len(documents),len(processed_docs),STEMMING,))
        processed_docs = pd.concat([prepro_doc,processed_docs],axis=0)
        with open(f"data/.preprocessed/{'s' if STEMMING else 'l'}_{year}_preprocessed.pickle","wb") as file:
            pickle.dump(prepro_doc, file)
    print("Documents preprocessed and saved")


    dictionary = model.create_dict(processed_docs, NUM_TOPICS, filter_n_most_freq=10)
    print("\nDictionary created:", dictionary)

    # print(sorted(dictionary.cfs.values(),reverse=True))

    corpus_bow = [dictionary.doc2bow(doc) for doc in processed_docs]

    tfidf = TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]

    lda_model_bow = model.run_model(LdaModel, corpus_bow, NUM_TOPICS, dictionary, save_file="bow")

    lda_model_tfidf = model.run_model(LdaModel, corpus_tfidf, NUM_TOPICS, dictionary, save_file="tfidf")

    for i, lda_model, corpus in zip(("bow","tfidf"), (lda_model_bow, lda_model_tfidf), (corpus_bow, corpus_tfidf)):
        model.print_topics(lda_model, output=f"{'s' if STEMMING else 'l'}_model_{i}_k{NUM_TOPICS}_from{FROM_YEAR}")
        model.classify(lda_model,corpus,i,STEMMING)

    # lda_model_bow.show_topics


if __name__ == "__main__":
    main()
