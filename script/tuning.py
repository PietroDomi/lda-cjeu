import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import script.model as model
from gensim.models import CoherenceModel, LdaMulticore

def compute_coherence_values(lda_model, corpus, dictionary, processed_docs, k, a, b):
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='u_mass')
    
    return coherence_model_lda.get_coherence()

FROM_YEAR = 2010
STEMMING = False

to_load = []
processed_docs = pd.Series(dtype=object)
for year in range(2015, FROM_YEAR-1, -1):
    try:
        prepro_docs = pickle.load(open(f"data/.preprocessed/{'s' if STEMMING else 'l'}_{year}_preprocessed.pickle","rb"))
        processed_docs = pd.concat([processed_docs,prepro_docs], axis=0)
    except:
        to_load.append(year)

NUM_TOPICS = 20

dictionary = model.create_dict(processed_docs, NUM_TOPICS, filter_n_most_freq=10)
print("\nDictionary created:", dictionary)

# print(sorted(dictionary.cfs.values(),reverse=True))

corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

grid = {}
grid['Validation_Set'] = {}

# Topics range
min_topics = 2
max_topics = 20
step_size = 2
topics_range = [5,7,10,15,20,50] #range(min_topics, max_topics, step_size)

# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

model_results = {#'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }

# Can take a long time to run
if True:
    pbar = tqdm.tqdm(total = len(topics_range) * len(alpha) * len(beta))
    
    # iterate through validation corpuses
        # iterate through number of topics
    for k in topics_range:
        # iterate through alpha values
        for a in alpha:
            # iterare through beta values
            for b in beta:
                # get the coherence score for the given parameters
                lda_model = LdaMulticore(corpus=corpus, num_topics=k, alpha=a, eta=b)
                cv = compute_coherence_values(lda_model=lda_model, corpus=corpus, processed_docs=processed_docs, dictionary=dictionary, 
                                                k=k, a=a, b=b)
                # Save the model results
                # model_results['Validation_Set'].append(corpus)
                model_results['Topics'].append(k)
                model_results['Alpha'].append(a)
                model_results['Beta'].append(b)
                model_results['Coherence'].append(cv)
                
                pbar.update(1)
    a = pd.DataFrame(model_results)
    a.sort_values("Coherence", ascending=False, inplace=True)
    a.to_csv('data/lda_tuning_results_5_50.csv', index=False)
    pbar.close()

results = pd.read_csv("data/lda_tuning_results_2_18.csv")
results2 = pd.read_csv("data/lda_tuning_results_5_50.csv")
results = pd.concat([results, results2], axis=0)
coh = results[results.Alpha == "symmetric"]
coh = coh[coh.Beta == "symmetric"]
coh.sort_values("Topics", inplace=True)
plt.plot(coh.Topics.drop(162), coh.Coherence.drop(162))
plt.xlabel("Num. of Topics")
plt.ylabel("Coherence")
plt.title("Coherence (UMass) when alpha=symmetric, beta=symmetric")
plt.savefig("fig/coherence.png")
plt.close()