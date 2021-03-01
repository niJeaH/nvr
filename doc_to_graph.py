import networkx as nx
import matplotlib.pyplot as plt
################################################################72limit79limit
class Doc_to_Graph:
    def __init__(self, direction = True, multi = False):

        if direction and multi :
            self.G = nx.MultiDiGraph()
        elif direction and not multi : 
            self.G = nx.DiGraph()
        elif not direction and not multi : 
            self.G = nx.Graph()
        else :
            self.G = nx.MultiGraph()

        self.node_list = []
        self.edge_list = []

    def doc_to_graph(self, sfdoc):

        for i, sent in enumerate(sfdoc.sentences):
            print("[Sentence {}]".format(i+1))
            #print("[Sentence {}]".format(i+1))
            stli=[]
            sted=[]
            for j,word in enumerate(sent.words): # process one sentences
                if word.lemma in dict(stli).keys():
                    dict(stli)[word.lemma]['weight'] += 1
                    if word.upos not in dict(stli)[word.lemma]['upos']:
                        dict(stli)[word.lemma]['upos'].append(word.upos)
                    if word.text not in dict(stli)[word.lemma]['text']:
                        dict(stli)[word.lemma]['text'].append(word.text)
                    if j not in dict(stli)[word.lemma]['sent_id']:
                        dict(stli)[word.lemma]['sent_id'].append(j)    
                    
                else:
                    stli.append((word.lemma,{'weight':1,'upos': [word.upos],
                    'text':[word.text], "sent_id" : [j]}))

                if 'tmp1' not in locals():
                    tmp1 = (word.lemma,{'weight':1,'upos': word.upos})
                else:
                    tmp2 = (tmp1[0],word.lemma,{'weight':1, 
                    'upos_pattern':[(tmp1[1]['upos'],word.upos)],
                    "sent_id":[j]})
                    
                    sted.append(tmp2)
                    if self.G.has_edge(tmp2[0],tmp2[1]):
                        prev_edge_data = self.G.get_edge_data(tmp2[0],tmp2[1])
                        if tmp2[2]['upos_pattern'][0] not in \
                            prev_edge_data['upos_pattern'] :
                            prev_edge_data['upos_pattern'].append(
                                tmp2[2]['upos_pattern'][0])

                        prev_edge_data['sent_id'].append(tmp2[2]['sent_id'][0])
                        prev_edge_data['weight'] += 1
                        tmp3 = [(tmp2[0],tmp2[1],prev_edge_data)]
                        self.G.add_edges_from(tmp3)
                    else : 
                        self.G.add_edges_from([tmp2])
                    tmp1 = (word.lemma,{'weight':1,'upos': word.upos})

            del(tmp1)
            self.node_list.append(stli)
            self.G.add_nodes_from(stli)

        return self.G

    def reset(self):
        self.G = nx.DiGraph()
        self.node_list = []
        self.edge_list = []

        