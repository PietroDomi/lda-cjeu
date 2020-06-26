import pandas as pd 

topics = pd.read_csv("data/output/l_model_tfidf_k16_from2000.csv", header=None)

topic_list = [[] for i in range(topics.shape[0])]

for i in range(topics.shape[0]):
    topic = topics[i]
    for j, word in enumerate(topic):
        topic_list[j].append(word)

for i, topic in enumerate(topic_list):
    print("\midrule "+str(i+1),end="")
    for tuple in topic:
        word = tuple.split("(")[1].split(")")[0].split(",")
        term = word[0][1:]
        term = term[:-1]
        print(" & ", term , " & " , float(word[1]),end=" \\\\ \n")
