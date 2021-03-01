import env_control
#env_control.save_env(filename="donetheG.pkl")
env_control.load_env(filename="donetheG.pkl") 
env_control.load_env(filename="node.pkl") 
from doc_to_graph import *
tagged[0].sentences[0].words[5]
brand_name = revs.brand.unique()
len(revs)
total_G = []
refs = nlp("this is sample text")
for j,bname in enumerate(brand_name):
    DTG = Doc_to_Graph()
    tmp_revs = tagged[revs['brand']== bname]
    for i in tmp_revs.index:
        if type(tmp_revs[i]) == type(refs):
            G = DTG.doc_to_graph(tmp_revs[i])
        else:
            G = tmp_revs[i]
    total_G.append(G)
None

nx.to_pandas_edgelist(total_G[0]).iloc[:37,]


###################################################################################################################
from networkx.algorithms import approximation
###################################################################################################################
from statistics import mean 

revs_all = pd.read_csv("C:\\Users\\HaeJinKim\\Desktop\\2018-2019\\Paper\\ACL\\AMR\\Amazone crawling data\\all_rev_2.csv",encoding = 'unicode_escape')
set1 = []

for i in range(len(total_G)):
    G = total_G[i]
    # # of nodes, n
    a1 = G.number_of_nodes()
    # # of edges, m
    a2 = G.number_of_edges()
    
    #degree, c
    a3 = mean(dict(G.degree()).values())
    # mean in degree, c1
    a4 = mean(dict(G.in_degree()).values())
    # mean out degree, c2
    a5 = mean(dict(G.out_degree()).values())
    # fraction of nodes in largest (weakly connected) component,S
    a6 = nx.is_strongly_connected(G)
    a7 = nx.is_weakly_connected(G)

    # mean geodasic distance, l
    a8 = nx.number_of_selfloops(G)

    # alpha, a
    # Clustering coefficient, c
    a9 = nx.average_clustering(G)

    # alternative clustering coefficent, Cws
    a10 = nx.transitivity(G)

    # degree correlation coefficient, r 
    a11 = nx.average_degree_connectivity(G)

    tmp1=[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11]
    set1.append(tmp1)




##############################################################################3


None
brand_name
DTG = Doc_to_Graph()
st_Sony=tagged[revs['brand']=='Sony']
range(len(st_Sony))
for i in range(len(st_Sony)):
    G_sony=DTG.doc_to_graph(st_Sony[i])
None
G_sony.adj

DTG2 = Doc_to_Graph()
st_Samsung=tagged[revs['brand']=='Samsung']
for i in st_Samsung.index:
    G_Samsung=DTG2.doc_to_graph(st_Samsung[i])
None

G = G_Samsung
nx.draw(G, with_label = True, font_weight ='bold', labels = True, node_size = 1)
plt.show()
len(G_Samsung.edges)
len(G_sony.edges)
len(G_Samsung.nodes)
type(G_Samsung.nodes())
type(G_Samsung.nodes)
node_samsung=list(G_Samsung.nodes())
node_sony=list(G_sony.nodes())

nodesAt5 = filter(lambda (n, d): d['upos_from'] == "VERB", G_Samsung.nodes(data=True))


node_samsung 
len(set(node_samsung) & set(node_sony))

######################################################################################################################################################
import env_control
env_control.save_env("all_revs.pkl")
env_control.load_env()
from doc_to_graph import *

import matplotlib.pyplot as plt
import pandas as pd
import re 
import string
import numpy as np
import stanfordnlp as sfnlp
import networkx as nx
from datetime  import datetime
#sfnlp.download('en')
nlp = sfnlp.Pipeline() # load pipeline

all_revs = pd.read_csv("C:\\Users\\HaeJinKim\\Desktop\\2018-2019\\Paper\\ACL\\AMR\\Amazone crawling data\\all_rev_2.csv",encoding = 'unicode_escape')
all_revs = all_revs.iloc[:,1:12]

product_name=all_revs.Product_name.unique()
brand_name = all_revs.brand.unique()
len(product_name)
len(brand_name)


set1[0]