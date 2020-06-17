import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get(from_year, show=False):
    data = pd.read_csv("data/Search Results 20200507.csv")

    celexs = sorted(data["CELEX number"].to_list())

    distr, _ = to_get_list(celexs)
    
    # tot = 0
    # print("Year","Cum. Sum\n")
    # for i in range(2015,2000,-1):
    #     tot += np.sum(np.array(distr)==i)
    #     print(i,tot)

    print()
    if show:
        plt.hist(distr,bins=(np.max(distr)-np.min(distr))+1)
        plt.title("Documents distribution through time")
        plt.xlabel("Year")
        plt.ylabel("Count")
        plt.show()

    distr, to_get = to_get_list(celexs, from_year, create_txt=True)

    with open("data_scraping/to_get.txt","w") as file:
        for c in sorted(to_get):
            file.writelines(c+"\n")


def to_get_list(celexs, from_year=2010, create_txt=False):

    to_get = set()
    years = np.arange(from_year,2021)
    distr = []
    for celex in celexs:
        distr.append(int(celex[1:5]))
        if create_txt and int(celex[1:5]) in years:
            if len(celex) > 11:
                celex = celex[:11]
                if celex not in to_get:
                    to_get.add(celex)
            else:
                to_get.add(celex)

    return distr, to_get