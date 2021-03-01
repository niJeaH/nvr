cd jamr
bash
python3

import dill
import networkx as nx
from networkx.algorithms import isomorphism
from datetime import datetime
import re

exm=get_edge_list_from_AMR(test)
sentence_print(test)

def sentence_print(AMR, opt = 0):
    if opt == 0 :
        for line in AMR:
            if (line.startswith("(")or line.startswith(' ') or line.startswith("# ::tok")):
                print(line)
    elif opt == 1:
        for line in AMR:
            if line.startswith("# ::tok"):
                print(line)
    else :
        if (line.startswith("(")or line.startswith(' ')):
            print(line)

def get_node_from_AMR(line):
    pat1 = re.compile('.\s\/\s')
    pat2 = re.compile('\"')
    if(line.startswith('(') or line.startswith(' ')):
        line = line.rstrip()
        if (re.search(pat1,line) == None and re.search(pat2, line) != None):
            stidx = re.search(pat2, line).span()[0]
        elif line.find('(')<0: stidx = line.find(')') -1
        else : stidx=line.find('(')+1
        node_subject = line[stidx:].replace(')','')
        return node_subject

def get_edge_list_from_AMR(lines):# input lines is a AMR parsed lines, dt3[,'AMR']
    edge_list=[]
    sentid = 0
    for line in lines:
        if line.startswith('('):
            p_list = [get_node_from_AMR(line)]
            prev_line = 0
        elif line.startswith(' '):
            if line.find(":") == prev_line + 6:
                node_from = p_list[len(p_list)-1]
            elif line.find(":") == prev_line:
                _=p_list.pop()
                node_from = p_list[len(p_list)-1]
            elif line.find(":") == prev_line - 6:
                _=p_list.pop()
                _=p_list.pop()
                node_from = p_list[len(p_list)-1]
            if line.find(":") > 0 :
                st_idx = line.find(":")+1
                sub_line = line[st_idx:]
                ed_idx = sub_line.find(" ")
                edge_attr = {'type' : sub_line[:ed_idx], 'st_idx' : sentid}
                node_to = get_node_from_AMR(line)
            edge_list.append((node_from,node_to,edge_attr))
            _ = p_list.append(get_node_from_AMR(line))
            prev_line = line.find(":")
        elif line.startswith("# ::tok") : 
            sentid += 1
    return edge_list

############################################# To attach the AMR to data frame
dill.load_session('batch1_IT.pkl')

df1 = dt1
iteration = 0
pds = []
for i in df1.AMR :
    G = nx.DiGraph()
    if i != None:
        edge_A=get_edge_list_from_AMR(i)
        _=G.add_edges_from(edge_A)
    _=pds.append(G)
    iteration += 1
    print('\t {0} th review added'.format(iteration), end = '\r')

get_edge_list_from_AMR(batch.AMR.iloc[0])
pds[0]
dt1['dg'] = pds
dt3['dg'] = pds
dt2['dg'] = pds

## Data stack 

batch_B = pd.concat([batch1,batch2,batch3])
batch_C = pd.concat([dt1,dt2,dt3])
batch_A = batch_B.set_index('X').join(batch_C.set_index('IDX'))
batch_A.to_pickle('./parsedAMR.pkl')
######################################################################

import pandas as pd #
import numpy as np#
import networkx as nx#
from datetime import timedelta#
from networkx.algorithms import isomorphism
from networkx.algorithms import approximation, isomorphism, dag
batch_A = pd.read_pickle('./parsedAMR.pkl')

## star rating and Price extraction
batch_A.rvstar = [float(i.replace(' out of 5 stars','')) for i in batch_A.rvstar]
batch_A.Price = [float(i.replace('$','')) for i in batch_A.Price]

#dill.dump_session('AMR_starting_point.pkl')
dill.load_session('AMR_starting_point.pkl')

## week ID generation 
batch = batch_A
batch['wkid']=[str(i.year)+'-'+i.strftime('%V') for i in batch_A.rv_date]
batch.groupby(['brand','wkid']).agg(['count']).to_csv('what.csv')




#:: graph trimmers
# 3PAPERS BEFORE. 
def get_node_concept(node):
    if node.find(" / ") > 0 :
        return node[node.find(" / ")+3:]
    else :
        return node.replace('\"',"")

def get_concept_graph(G):
    e_list = [(get_node_concept(u),get_node_concept(v), e) for u,v,e in G.edges(data=True)]
    G_1 = nx.DiGraph(e_list)
    return G_1

def subgraph_per_sentence(G, option = 0):
    edges = []
    for i in set([e['st_idx'] for _,_,e in G.edges(data=True)]):
        sub_sent = [(u,v,e) for u,v,e in G.edges(data=True) if e['st_idx'] == i]
        if len(sub_sent) > 0 :
            edges.append(sub_sent)
    if option == 1 :
        dags = []
        for el in edges:
            dags.append(nx.DiGraph(el))
        edges = dags
    
    return edges

#:: triad detections 
def find_corerole_triad(G):
    core_role = ['ARG0','ARG1','ARG2','ARG3','ARG4','ARG5','ARG0-of','ARG1-of','polarity']
    triad = []
    triad2 = []
    for i in set([e['st_idx'] for u,v,e in G.edges(data=True)]):
        core_edges = [(get_node_concept(u),e['type'],get_node_concept(v)) for u,v,e in G.edges(data=True) if e['type'] in core_role and e['st_idx'] == i ]
        for u1,e1,v1 in core_edges:
            tmp = [(u1,e1,v1,e,v) for u,e,v in core_edges if u == v1]
            tmp2 = [(u1,v1,e1,u,v,e) for u,e,v in core_edges if (u == v1 and u1 != v) or (v1 == v and u1 != u) or (u1 == u and v1 != v)]
            if len(tmp)> 1 : 
                triad.extend(tmp)
                triad2.extend(tmp2)
    return triad, triad2

def find_triad(G):
    triad = []
    triad2 = []
    for i in set([e['st_idx'] for u,v,e in G.edges(data=True)]):
        core_edges = [(get_node_concept(u),e['type'],get_node_concept(v)) for u,v,e in G.edges(data=True)]
        for u1,e1,v1 in core_edges:
            tmp = [(u1,e1,v1,e,v) for u,e,v in core_edges if u == v1]
            tmp2 = [(u1,v1,e1,u,v,e) for u,e,v in core_edges if (u == v1 and u1 != v) or (v1 == v and u1 != u) or (u1 == u and v1 != v)]
            if len(tmp)> 1 : 
                triad.extend(tmp)
                triad2.extend(tmp2)
    return  triad,triad2

def find_corerole_pair(G):  
    core_role = ['ARG0','ARG1','ARG2','ARG3','ARG4','ARG5','ARG0-of','ARG1-of','polarity']
    pair = []
    for i in set([e['st_idx'] for u,v,e in G.edges(data=True)]):
        core_edges = [(get_node_concept(u),e['type'],get_node_concept(v)) for u,v,e in G.edges(data=True) if e['type'] in core_role and e['st_idx'] == i ]
        pair.extend(core_edges)
    return pair

######################################
#review sorting by week 
#:: Triad analysis
batch.brand[(batch.rv_date >= '2016-07-30') & (batch.rv_date < '2016-08-07')]
h1=(batch.rv_date >= '2016-11') & (batch.rv_date < '2016-12')

core_triad = []
timevalue = datetime.now()
for i in batch.dg:
    if type(i) != type(0.0):
        _,tmp = find_corerole_triad(i)
        core_triad.extend(tmp)
    print('\t{}'.format(datetime.now()-timevalue), end = '\r')

len(core_triad) # 14471, 20179
len(set(core_triad)) # 13590, 19127
core_triad_set=set(core_triad) # 13590

multioccured_ctr =[]
for i in core_triad_set:
    if core_triad.count(i) > 1:
        multioccured_ctr.append((i,core_triad.count(i)))
        print('\t {0} had occured {1} \t'.format(i,core_triad.count(i)), end ='\r')

len([u for u,i in multioccured_ctr if i > 1])

from networkx.algorithms import dag
triad_lg = []
timevalue = datetime.now()
for i in batch.dg[batch.brand =='LG']:
    if type(i) != type(0.0):
        _,tmp = find_triad(i)
        triad_lg.extend(tmp)
    print('\t Food Morning!{}'.format(datetime.now()-timevalue), end = '\r')

triad_hw = []
timevalue = datetime.now()
for i in batch.dg[batch.brand != 'LG']:
    if type(i) != type(0.0):
        _,tmp = find_triad(i)
        triad_hw.extend(tmp)
    print('\t Food Morning!{}'.format(datetime.now()-timevalue), end = '\r')

len(triad_lg)
len(triad_hw)


triad_lg_A = []
timevalue = datetime.now()
for i in batch.dg[(batch.brand =='LG') & H[0]]:
    if type(i) != type(0.0):
        _,tmp = find_triad(i)
        triad_lg_A.extend(tmp)
    print('\t Food Morning!{}'.format(datetime.now()-timevalue), end = '\r')

triad_lg_B = []
timevalue = datetime.now()
for i in batch.dg[(batch.brand =='LG') & H[1]]:
    if type(i) != type(0.0):
        _,tmp = find_triad(i)
        triad_lg_B.extend(tmp)
    print('\t Food Morning!{}'.format(datetime.now()-timevalue), end = '\r')

len(set(triad_lg_A).intersection(set(triad_lg_B))) #8996
len(set(triad_lg_A)-set(triad_lg_B)) #36430
len(set(triad_lg_B)-set(triad_lg_A)) #56478
len(set(triad_lg_B))# A: 45426, B : 65474

triad_hw_A = []
timevalue = datetime.now()
for i in batch.dg[(batch.brand != 'LG') & H[0]]:
    if type(i) != type(0.0):
        _,tmp = find_triad(i)
        triad_hw_A.extend(tmp)
    print('\t Food Morning!{}'.format(datetime.now()-timevalue), end = '\r')

triad_hw_B = []
timevalue = datetime.now()
for i in batch.dg[(batch.brand != 'LG')& H[1]]:
    if type(i) != type(0.0):
        _,tmp = find_triad(i)
        triad_hw_B.extend(tmp)
    print('\t Food Morning!{}'.format(datetime.now()-timevalue), end = '\r')

len(set(triad_hw_A).intersection(set(triad_hw_B))) #15402
len(set(triad_hw_A)-set(triad_hw_B)) #89539
len(set(triad_hw_B)-set(triad_hw_A)) #59447
len(set(triad_hw_A))# A: 104941, B : 74849


from collections import Counter
import matplotlib.pyplot as plt

clg = Counter(triad_lg)
chw = Counter(triad_hw)
common_triad=list(set(triad_lg).intersection(set(triad_hw)))
#common_triad_df = pd.DataFrame({'triad':list(common_triad), 'lg':[dict(clg).get(tri) for tri in common_triad], 'hw':[dict(chw).get(tri) for tri in common_triad]})
common_triad_df
len(common_triad)
del triad_lg, triad_hw

dill.load_session('paths_man.pkl')

#dill.dump_session('triad_analysis.pkl')
dill.load_session('triad_analysis.pkl')

#:: split by duration 
H=[((batch.rv_date >= '2016-10') & (batch.rv_date < '2017-01')),\
((batch.rv_date >= '2017-01') & (batch.rv_date < '2017-04')),
((batch.rv_date >= '2017-04') & (batch.rv_date < '2017-07')),
((batch.rv_date >= '2017-07') & (batch.rv_date < '2017-10')),
((batch.rv_date >= '2017-10') & (batch.rv_date < '2018-01'))]

L=[((batch.rv_date >= '2017-01') & (batch.rv_date < '2017-07')),\
((batch.rv_date >= '2017-06') & (batch.rv_date < '2017-12'))]

qt_idx = ['2017-A','2017-B']
qt_idx = ['2016-Q4','2017-Q1','2017-Q2','2017-Q3','2017-Q4']
triad_by_q = pd.DataFrame({'triad':list(common_triad)})
for i,h in enumerate(H):
    tmp_triad_lg = []
    for g in batch.dg[(batch.brand =='LG') & h]:
        if type(g) != type(0.0):
            _,tmp = find_triad(g)
            tmp_triad_lg.extend(tmp)
    tmp_triad_hw = []
    for g in batch.dg[(batch.brand !='LG') & h]:
        if type(g) != type(0.0):
            _,tmp = find_triad(g)
            tmp_triad_hw.extend(tmp)        
    #tmp_triad_hw = [find_triad(g) for g in batch.dg[(batch.brand !='LG') & h] if type(g) != type(0.0)]
    clg = Counter(tmp_triad_lg)
    chw = Counter(tmp_triad_hw)
    triad_by_q['LG '+qt_idx[i]] = pd.Series([dict(clg).get(tri) for tri in common_triad])
    triad_by_q['HW '+qt_idx[i]] = pd.Series([dict(chw).get(tri) for tri in common_triad])
    print('\t{0} just had processed in {1}'.format(i,datetime.now()), end = 'r')

triad_by_q.to_csv('triad.csv')


#:: information score analysis
########## calculate the node score of tf-idf
allnodes = []
for i in batch.dg:
    if type(i) != type(0.0):
        allnodes.extend([get_node_concept(h) for h in list(i.nodes)])

dd = {x:allnodes.count(x) for x in allnodes}
allnode = pd.DataFrame({'node': list(dd.keys()),'freq':list(dd.values())})
allnode.sort_values(by=['freq'], ascending = False)
allnode['freq_score']=allnode.freq/allnode.freq.max()*0.5 +0.5
allnode['doc_freq'] = allnode.freq/allnode.freq
for l in allnode.index:
    for g in batch.dg:
       if allnode.node[l] in [get_node_concept(h) for h in list(g.nodes)]:
           allnode.doc_freq[l] +=1
    print('\t processing {} node'.format(l), end ='\r')

allnode['doc_score'] = np.log(len(batch)/allnode.doc_freq)
allnode['tf-idf'] = allnode.freq_score*allnode.doc_score
allnode.to_csv('node_freq.csv')
## get the core_triad_lg
## assign core_triad number from lg 
def get_informative_score(triad):
    u1,_,v1,u2,_,v2 = (triad)
    return sum(allnode['tf-idf'][allnode.node.isin([u1,v1,u2,v2])])

##################################################################################
#  get the directed Acyclic subgraph.
#:: path analysis
def get_edge_path(G):
    paths = [h for h in dag.root_to_leaf_paths(G)]
    new_paths = []
    #toe =[]
    for elm in paths : # elm is each path in all paths of node
        #toe.append([(elm[el],elm[el+1]) for el in range(len(elm)-1)])
        toe = [(elm[el],elm[el+1]) for el in range(len(elm)-1)] #get the edge_tuple
        h = []
        for i in toe : # get the edge linking u,v
            h.extend([e['type'] for u,v,e in G.edges(data=True) if (u,v) == i])
        elm = [esn(node) for node in elm] # trim the amr node to amr concept
        #insert the edge h between nodes
        _=[elm.insert(k,h[int(k/2)]) for k in [ix for ix in range(len(toe)+len(elm)) if ix%2 !=0]]
        new_paths.append(tuple(elm))
    return new_paths

def esn(node):
    return get_node_concept(node)

lg_path = []
for di, g in enumerate(batch.dg[batch.brand =='LG']):
    H=subgraph_per_sentence(g,option=1)
    for i,item in enumerate(H):
        lg_path.extend(get_edge_path(item))
        print('\t Good Morning Haejin! ive extract {0}th paths from {1}'.format(i,di), end = '\r')

hw_path = []
for di, g in enumerate(batch.dg[batch.brand !='LG']):
    H=subgraph_per_sentence(g,option=1)
    for i,item in enumerate(H):
        hw_path.extend(get_edge_path(item))
        print('\t Good Morning Haejin! ive extract {0}th paths from {1}'.format(i,di), end = '\r')


common_paths = set(lg_path).intersection(hw_path)
common_lg = [lg_path.count(i) for i in common_paths]
common_hw = [hw_path.count(i) for i in common_paths]
path_count_lg = [lg_path.count(i) for i in lg_path]
path_count_hw =[hw_path.count(i) for i in hw_path]
lgt=[i > 10 for i in path_count_lg]
hwt=[i > 10 for i in path_count_hw]

len(common_paths)
var1=pd.DataFrame({'path':list(common_paths),'lg':common_lg, 'hw':common_hw})
var1.to_csv('congrats.csv')


lg_path_A = []
for di, g in enumerate(batch.dg[(batch.brand =='LG')& L[0]]):
    H=subgraph_per_sentence(g,option=1)
    for i,item in enumerate(H):
        lg_path_A.extend(get_edge_path(item))
        print('\t Good Morning Haejin! ive extract {0}th paths from {1}'.format(i,di), end = '\r')

lg_path_B = []
for di, g in enumerate(batch.dg[(batch.brand =='LG') & L[1]]):
    H=subgraph_per_sentence(g,option=1)
    for i,item in enumerate(H):
        lg_path_B.extend(get_edge_path(item))
        print('\t Good Morning Haejin! ive extract {0}th paths from {1}'.format(i,di), end = '\r')

len(set(lg_path_A).intersection(set(lg_path_B))) # 920
lg_common = set(lg_path_A).intersection(set(lg_path_B))

hw_path_A = []
for di, g in enumerate(batch.dg[(batch.brand !='LG')& L[0]]):
    H=subgraph_per_sentence(g,option=1)
    for i,item in enumerate(H):
        hw_path_A.extend(get_edge_path(item))
        print('\t Good Morning Haejin! ive extract {0}th paths from {1}'.format(i,di), end = '\r')

hw_path_B = []
for di, g in enumerate(batch.dg[(batch.brand !='LG') & L[1]]):
    H=subgraph_per_sentence(g,option=1)
    for i,item in enumerate(H):
        hw_path_B.extend(get_edge_path(item))
        print('\t Good Morning Haejin! ive extract {0}th paths from {1}'.format(i,di), end = '\r')

len(set(hw_path_A).intersection(set(hw_path_B))) # 1510
hw_common = set(hw_path_A).intersection(set(hw_path_B))

len(hw_common.intersection(lg_common))

path_by_q = pd.DataFrame({'triad':list(common_paths)})
for i,h in enumerate(H):
    tmp_path_lg = []
    for g in batch.dg[(batch.brand =='LG') & h]:
        if type(g) != type(0.0):
            sub_dag = subgraph_per_sentence(g, option=1)
            for j,item in enumerate(sub_dag):
                tmp_path_lg.extend(get_edge_path(item))
    tmp_path_hw = []
    for g in batch.dg[(batch.brand !='LG') & h]:
        if type(g) != type(0.0):
            sub_dag = subgraph_per_sentence(g, option=1)
            for j,item in enumerate(sub_dag):
                tmp_path_hw.extend(get_edge_path(item))
    #tmp_triad_hw = [find_triad(g) for g in batch.dg[(batch.brand !='LG') & h] if type(g) != type(0.0)]
    clg = Counter(tmp_path_lg)
    chw = Counter(tmp_path_hw)
    path_by_q['LG '+qt_idx[i]] = pd.Series([dict(clg).get(tri) for tri in common_paths])
    path_by_q['HW '+qt_idx[i]] = pd.Series([dict(chw).get(tri) for tri in common_paths])
    print('\t{0} just had processed in {1}'.format(i,datetime.now()), end = '\r')

path_by_q.to_csv('path_by_quarter.csv')


import pandas as pd
#dill.dump_session('paths_man.pkl')

qt_idx = ['2016-Q4','2017-Q1','2017-Q2','2017-Q3','2017-Q4']
path_by_q = pd.DataFrame({'triad':list(common_paths)})
for i,h in enumerate(H):
    tmp_path_lg = []
    for g in batch.dg[(batch.brand =='LG') & h]:
        if type(g) != type(0.0):
            sub_dag = subgraph_per_sentence(g, option=1)
            for j,item in enumerate(sub_dag):
                tmp_path_lg.extend(get_edge_path(item))
    tmp_path_hw = []
    for g in batch.dg[(batch.brand !='LG') & h]:
        if type(g) != type(0.0):
            sub_dag = subgraph_per_sentence(g, option=1)
            for j,item in enumerate(sub_dag):
                tmp_path_hw.extend(get_edge_path(item))
    #tmp_triad_hw = [find_triad(g) for g in batch.dg[(batch.brand !='LG') & h] if type(g) != type(0.0)]
    clg = Counter(tmp_path_lg)
    chw = Counter(tmp_path_hw)
    path_by_q['LG '+qt_idx[i]] = pd.Series([dict(clg).get(tri) for tri in common_paths])
    path_by_q['HW '+qt_idx[i]] = pd.Series([dict(chw).get(tri) for tri in common_paths])
    print('\t{0} just had processed in {1}'.format(i,datetime.now()), end = '\r')

path_by_q.to_csv('path_by_quarter.csv')

all_path = set(lg_path).union(set(hw_path))
('i','ARG0-of') in list(all_path)[0]
gen = (x for x in all_path if ('battery' in x) or ('screen' in x)\
     or ('camera' in x) or ('price-01' in x))
len(all_path)
focus_path = list(gen)
features = ['battery','screen','camera','price-01']
[features[0] in x for x in list(tmp_path_lg)]

feature_count_df = pd.DataFrame({'Feature':focus_path})
for i,h in enumerate(H):
    tmp_path_lg = []
    for g in batch.dg[(batch.brand =='LG') & h]:
        if type(g) != type(0.0):
            sub_dag = subgraph_per_sentence(g, option=1)
            for j,item in enumerate(sub_dag):
                tmp_path_lg.extend(get_edge_path(item))
    tmp_path_hw = []
    for g in batch.dg[(batch.brand !='LG') & h]:
        if type(g) != type(0.0):
            sub_dag = subgraph_per_sentence(g, option=1)
            for j,item in enumerate(sub_dag):
                tmp_path_hw.extend(get_edge_path(item))
    clg = Counter(tmp_path_lg)
    chw = Counter(tmp_path_hw)
    feature_count_df['LG '+qt_idx[i]] = pd.Series([dict(clg).get(tri) for tri in focus_path])
    feature_count_df['HW '+qt_idx[i]] = pd.Series([dict(chw).get(tri) for tri in focus_path])
    print('\t{0} just had processed in {1}'.format(i,datetime.now()), end = '\r')

feature_count_df.to_csv('focut_features.csv')

path_all = set(lg_path).union(set(hw_path))

get_edge_path(H[0])

H[0].edges(data=True)


import dill
#dill.dump_session('infworth_starting.pkl')
dill.load_session('infworth_starting.pkl')
dill.load_session('triad_starting.pkl')
dill.load_session('paths_man.pkl')
dill.dump_session('bef.pkl')
len(core_triad_lg)
len(core_triad_hw)
common_core = set(core_triad_lg).intersection(set(core_triad_hw))
common_core
LG_informative = [get_informative_score(i) for i in core_triad_lg]
Hw_informative = [get_informative_score(i) for i in core_triad_hw]
sum(LG_informative)
sum(Hw_informative)


infscore=[]
for dg in batch.dg:
    if type(dg) != type(0.0):
        _,tmp = find_corerole_triad(dg)
        infscore.append(sum([get_informative_score(i) for i in tmp]))

se = pd.Series(infscore)
batch['informative_score']= se.values

save = batch[['brand','Product_name','Price','rvstar','rv_date','informative_score']]
list(i.strftime('%Y-')+i.strftime('%V') for i in save.rv_date)
(i.strftime('%Y-%V') for i in save.rv_date)

my_list = (i.strftime('%Y-%V') for i in save.rv_date)
se = pd.Series(list(my_list))
save['wkid'] = se.values


marketshare
marketshare = pd.read_excel('marketshare.xlsx',index_col=0)
results = save.merge(marketshare[['wkid','LG','Huawei']], on='wkid')
results['ms'] = results.LG +results.Huawei

results['ms'][results.brand != 'LG'] = results['LG'][results.brand == 'LG']
results['ms'][results.brand == 'LG'] = results['Huawei'][results.brand != 'LG']

results.to_csv('results.csv')

dill.dump_session('yap.pkl')
dill.load_session('yap.pkl')
## group by Brand 
## group by Week 


############################## for drawing

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, nudge=1.3):
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')
    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))
    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children)*1.5
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx*nudge
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

G=nx.DiGraph(exm)
G1 = subgraph_per_sentence(G, option = 1)
pos = hierarchy_pos(G1[1], width = 1,vert_gap = 0.05, nudge =0.8, xcenter = 1)
nx.draw(G1[1], pos = pos, with_labels = True, node_shape = 'o', alpha = 0.8, node_color ='#ffffff', font_size=15)
_=nx.draw_networkx_edge_labels(G1[1],pos ,edge_labels = nx.get_edge_attributes(G1[1],'type'), font_size = 15)
plt.savefig('example12.png')
plt.clf()

pos = hierarchy_pos(H[0], width = 18,vert_gap = 0.05, nudge = 0.8, xcenter=9)
pos['c / cause-01'] = (12,0)
pos['m / misrepresentation'] = (11, -0.05)
pos['w / week'] = (17.71875,-0.18)
plt.clf()
nx.draw(H[0], pos = pos, with_labels = True, node_shape = 'o', alpha = 0.8, node_color ='#ffffff', font_size=10)
_=nx.draw_networkx_edge_labels(H[0],pos ,edge_labels = nx.get_edge_attributes(H[0],'type'), font_size = 10)
plt.savefig('example2.png')

batch.columns
np.mean(batch.rvstar[batch.brand=="LG"])
np.mean(batch.rvstar[batch.brand!="LG"])
np.mean(batch.Price[batch.brand=="LG"])
np.mean(batch.Price[batch.brand!="LG"])
np.mean(batch.informative_score[batch.brand=="LG"])
np.mean(batch.informative_score[batch.brand!="LG"])
len(set(hw_path))
len(set(lg_path))
len(common_paths)
np.std(batch.rvstar[batch.brand=="LG"])
np.std(batch.rvstar[batch.brand!="LG"])
np.std(batch.Price[batch.brand=="LG"])
np.std(batch.Price[batch.brand!="LG"])
np.std(batch.informative_score[batch.brand=="LG"])
np.std(batch.informative_score[batch.brand!="LG"])

# 17
batch[batch.review.str.contains("return")]
ss=batch.review.str.contains("return")
s2 = batch.index[ss]
lv = s2[batch.brand[s2] =="LG"]
len(lv)#4
sentence_print(batch.AMR[lv[10]],opt = 0)
_,tmp=find_triad(batch.dg[lv[10]])
H = subgraph_per_sentence(batch.dg[lv[10]], option = 1)
print('\r\n{0}\r\n{1}\r\n'.format(get_edge_path(H[0])[0],get_edge_path(H[5])[1]))

triad2 = []
core_edges = [(get_node_concept(u),e['type'],get_node_concept(v)) for u,v,e in H[5].edges(data=True)]
for u1,e1,v1 in core_edges:
    tmp2 = [(u1,v1,e1,u,v,e) for u,e,v in core_edges if (u == v1 and u1 != v) or (v1 == v and u1 != u) or (u1 == u and v1 != v)]
    if len(tmp)> 1 : 
        triad2.extend(tmp2)

_=[print('{0}'.format(t)) for t in set(triad2)]

pos = nx.spring_layout(H[5])
pos = nx.planar_layout(H[5])
pos = nx.shell_layout(H[5])
pos = nx.circular_layout(H[5])
pos = nx.random_layout(H[5])
