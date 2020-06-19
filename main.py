import pandas as pd
from script import model, explore, html_converter
import os, pickle, argparse, json
from gensim.models import TfidfModel, LdaModel


def main():
    parser = argparse.ArgumentParser(description="LDA")
    parser.add_argument("--year", type=int, help=f"what year you would like to analyse the data from (year, corpus size): \n(2015, 7), (2014, 113), (2013, 132), (2012, 159), (2011, 242), (2010, 612), (2009, 1090), (2008, 1291), (2007, 1431), (2006, 1720), (2005, 2162), (2004, 2561), (2003, 2983), (2002, 3349), (2001, 3795)", default=2010)
    parser.add_argument("--k", type=int, metavar="NUM_TOPICS", help="number of topics for LDA", default=5)
    parser.add_argument("--plot", type=bool, metavar="SHOW DISTR PLOT", help="displays the distribution of documents through time", default=False)

    args = parser.parse_args()
    NUM_TOPICS = args.k

    if os.path.exists("old/last_run.txt"):
        with open("old/last_run.txt","r") as file:
            args_old = json.load(file)
            oldest_year = args_old['year']
            to_collect = (args.year < oldest_year)
        if to_collect:
            with open("old/last_run.txt","w") as file:
                d = args.__dict__
                d['year'] = oldest_year if oldest_year <= args.year else args.year
                json.dump(d, file, indent=2)
    else:
        with open("old/last_run.txt","w") as file:
            json.dump(args.__dict__, file, indent=2)
            to_collect = True

#TODO: invert order of checks
    print("\nChecking for preprocessed documents...")

    to_load = []
    processed_docs = pd.Series(dtype=object)
    for year in range(2015, args.year-1, -1):
        try:
            prepro_docs = pickle.load(open(f"data/.preprocessed/{year}_preprocessed.pickle","rb"))
            processed_docs = pd.concat([processed_docs,prepro_docs], axis=0)
        except:
            to_load.append(year)
    print("\nDocuments Loaded")
    

    if os.path.exists("data/converted") and int(os.listdir("data/converted")[0][1:5]) <= args.year:
        print("\nConverting data from html to txt...")
        NUM_DOCS = html_converter.convert()

    elif os.path.exists("data_scraping/data_html") and int(os.listdir("data_scraping/data_html")[0][1:5]) > args.year:
        print("\nCollecting data...")
        explore.get(from_year=args.year, show=args.plot)
        os.system("cd data_scraping && scrapy crawl celex")
        print("\n\nData Collected")
        NUM_DOCS = html_converter.convert()
        print(f"\nData converted: {NUM_DOCS} total documents")
        with open(".gitignore","a") as file:
            celexs = open("data_scraping/to_get.txt","r").readlines()
            for celex in celexs:
                file.writelines("data_scraping/data_html/"+celex[:-1]+".html\n")
                file.writelines("data/converted/"+celex[:-1]+".txt\n")

    NUM_DOCS = html_converter.convert()

    for year in sorted(to_load, reverse=True):
        print(f"\nProcesssing documents for {year}...")
        documents = pd.Series(model.load_docs(year, dir="data/converted/"))
        processed_docs = pd.concat([documents.apply(func=model.preprocess, args=(len(documents),len(processed_docs),)),processed_docs],axis=0)
        with open(f"data/.preprocessed/{year}_preprocessed.pickle","wb") as file:
            pickle.dump(processed_docs, file)
    print("Documents preprocessed and saved")


    dictionary = model.create_dict(processed_docs, NUM_TOPICS, filter_n_most_freq=10)
    print("\nDictionary created:", dictionary)

    # print(sorted(dictionary.cfs.values(),reverse=True))

    corpus_bow = [dictionary.doc2bow(doc) for doc in processed_docs]

    tfidf = TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]

    lda_model_bow = model.run_model(LdaModel, corpus_bow, NUM_TOPICS, dictionary, save_file="bow")

    lda_model_tfidf = model.run_model(LdaModel, corpus_tfidf, NUM_TOPICS, dictionary, save_file="tfidf")

    for i, (lda_model, corpus) in enumerate(zip((lda_model_bow, lda_model_tfidf), (corpus_bow, corpus_tfidf))):
        model.print_topics(lda_model, output=f"model_{i}_k{args.k}_from{args.year}")
        model.classify(lda_model,corpus,i)

    # lda_model_bow.show_topics


if __name__ == "__main__":
    main()
