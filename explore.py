#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/Search Results 20200507.csv")

# %%
celexs = sorted(data["CELEX number"].to_list())

# %%
to_get = set()
years = np.arange(2010,2016)
distr = []
for celex in celexs:
    distr.append(int(celex[1:5]))
    if int(celex[1:5]) in years:
        if len(celex) > 11:
            celex = celex[:11]
            if celex not in to_get:
                to_get.add(celex)
        else:
            to_get.add(celex)

# %%
plt.hist(distr,bins=(np.max(distr)-np.min(distr))+1)

# %%
tot = 0
for i in range(2015,2000,-1):
    tot += np.sum(np.array(distr)==i)
    print(i,tot)

# %%
with open("data_scraping/to_get.txt","w") as file:
    for c in sorted(to_get):
        file.writelines(c+"\n")

# %%
