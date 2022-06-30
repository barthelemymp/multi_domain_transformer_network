import networkx as nx
import pandas as pd
from itertools import *

###
from collections import defaultdict
def def_value():
    return False
count=0
# Defining the dict
d = defaultdict(def_value)
with open("unirefname.txt") as topo_file:
    for line in topo_file:
        protname = line.split(" ")[0].split("_")[1]
        d[protname]  = True
        count+=1
        if count%1000==0:
            print(count)
        
        
def def_valuegraph():
    return []


dgraph = defaultdict(def_valuegraph)

count=0
#with open("uniprot_reg_full.txt") as topo_file:
with open("uniprot_reg_full.txt") as topo_file:
    for line in topo_file:
        count+=1
        linesplit = line.split("\t")
        prot = linesplit[2]
        fam = linesplit[1]
        if d[prot]:
            dgraph[prot].append(fam)
        if count%1000==0:
            print(count)
            
  
            
### Make graph
G = nx.Graph()
count = 0
for prot in dgraph:
    count+=1
    for pfam in dgraph[prot]:
        if pfam in G.nodes:
            G.nodes[pfam]["count"] +=1
        else:
            G.add_node(pfam, count=1)
    for pair in combinations(dgraph[prot],2):
        if pair in G.edges:
            G.edges[pair]["weight"]+=1
        else:
            G.add_edge(*pair, weight=1)
    if count%1000==0:
        print(count)
        
### save load graph

nx.write_gml(G, "/home/barthelemy/fullUniref90.gml")


G = nx.read_gml("/home/barthelemy/fullUniref90.gml")

#filtering

G = nx.read_gml("/home/barthelemy/fullUniref90.gml")


node_weights = nx.get_node_attributes(G,'count')
G.remove_nodes_from((e for e, w in node_weights.items() if w < 3000))
edge_weights = nx.get_edge_attributes(G,'weight')
G.remove_edges_from((e for e, w in edge_weights.items() if w < 3000))
nx.write_gml(G, "/home/barthelemy/filterUniref90.gml")



df = pd.read_csv("pdbtracker.csv")
name = df["name"].unique()
edge_filter = nx.get_edge_attributes(G,'weight')
for edge in edge_filter:
    edge_filter[edge] = False
for nn in name:
    listfam = nn.split("_")
    fam1 = listfam[0]
    fam2 = listfam[1]
    pair1 = (fam1, fam2)
    pair2 = (fam2, fam1)
    if pair1 in edge_filter:
        edge_filter[pair1] = True
    if pair2 in edge_filter:
        edge_filter[pair2] = True
        
        

edge_weights = nx.get_edge_attributes(G,'weight')
G.remove_edges_from((e for e, w in edge_weights.items() if edge_filter[e]==False))
nx.write_gml(G, "/home/barthelemy/filterUniref90_contacts.gml")
        
G = nx.read_gml("/home/barthelemy/filterUniref90_contacts.gml")




CliqueGen =nx.find_cliques(G)
for clique in CliqueGen:
    if len(clique)>4:
        print(clique)

['PF00047', 'PF13927', 'PF07679', 'PF13895', 'PF08205']
['PF00047', 'PF13927', 'PF07679', 'PF07686', 'PF00041']
['PF00047', 'PF13927', 'PF07679', 'PF07686', 'PF08205']
['PF00047', 'PF13927', 'PF07679', 'PF07686', 'PF13855']
['PF01315', 'PF01799', 'PF00111', 'PF00941', 'PF02738']
['PF01799', 'PF00111', 'PF03450', 'PF00941', 'PF02738']