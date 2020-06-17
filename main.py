import pandas as pd
from script import model, explore, html_converter
import os, pickle, argparse, json
from gensim.models import TfidfModel, LdaModel


def main():
    parser = argparse.ArgumentParser(description="LDA")
    parser.add_argument("--year", type=int, help=f"what year you would like to analyse the data from (y, cumsum) {[()]}", default=2010)
    parser.add_argument("--k", metavar="NUM_TOPICS", help="number of topics for LDA", default=5)
    parser.add_argument("--plot", metavar="SHOW DISTR PLOT", help="displays the distribution of documents through time", default=False)

    args = parser.parse_args()

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


    if to_collect:
        print("Data collection:")
        explore.get(from_year=args.year, show=args.plot)
        os.system("cd data_scraping && scrapy crawl celex")
        print("\n\nData Collected")
        tot_docs = html_converter.convert()
        print(f"\nData converted: {tot_docs} total documents")
    elif int(os.listdir("data/converted")[0][1:5]) > args.year:
        print("\nConverting data from html to txt...")
        html_converter.convert()
    else:
        print("\nData already collected")        

    documents = pd.DataFrame()
    documents["text"] = model.load_docs(args.year, dir="data/converted/")
    print("\nDocuments Loaded")

    NUM_TOPICS = args.k
    NUM_DOCS = len(documents.text)

    print("\nChecking for preprocessed documents...")
    if os.path.exists(f"data/from_{args.year}_preprocessed.pickle"):
        with open(f"data/from_{args.year}_preprocessed.pickle","rb") as file:
            processed_docs = pickle.load(file)
        print("\nDocuments preprocessed loaded")
    else:
        print("Preprocesssing documents...")
        processed_docs = documents.text.apply(func=model.preprocess, args=(NUM_DOCS,))
        with open(f"data/from_{args.year}_preprocessed.pickle","wb") as file:
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

    for lda_model, corpus in zip((lda_model_bow, lda_model_tfidf), (corpus_bow, corpus_tfidf)):
        model.print_topics(lda_model)
        model.classify(lda_model,corpus)

    # lda_model_bow.show_topics


if __name__ == "__main__":
    main()
