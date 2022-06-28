import networkx as nx
import pandas as pd
import numpy as np
from itertools import *
from src.ProteinTransformer import *
from src.ProteinsDataset import *
import pickle

### Load the Uniref Graphs
# G = nx.read_gml("/Data/PFAM/fullUniref90.gml")
def def_value():
    return False

file_to_read = open("/Data/PFAM/unirefname.pkl", "rb")
uniref_dict = pickle.load(file_to_read)

def def_valuegraph():
    return []

file_to_read = open("/Data/PFAM/unirefarchitecture.pkl", "rb")
uniref_archi = pickle.load(file_to_read)


def myfilter(protname):
    mask=[]
    for prot in protname:
        if uniref_dict[prot]==False:
            mask.append(False)
        elif len(uniref_archi[prot])<2:
            mask.append(False)
        else:
            mask.append(True)
    return np.array(mask)
            


pmsa1 = ProteinMSA("/Data/PFAM/PF13857", onehot=False, protfilter =myfilter)
pmsa2 = ProteinMSA("/Data/PFAM/PF00023", onehot=False, protfilter =myfilter)
pmsa3 = ProteinMSA("/Data/PFAM/PF12796", onehot=False, protfilter =myfilter)
pmsa4 = ProteinMSA("/Data/PFAM/PF00069", onehot=False, protfilter =myfilter)


pathClique_list = ["/Data/PFAM/PF13857", "/Data/PFAM/PF00023", "/Data/PFAM/PF12796", "/Data/PFAM/PF00069"]#, "PF12796_rp35.txt"]
NameClique_list= ["PF13857", "PF00023", "PF12796", "PF00069"]

pnd = ProteinNetworkDataset( pathClique_list, NameClique_list,  mapstring="-ACDEFGHIKLMNPQRSTVWY", transform=None, device=None, batch_first=False, returnIndex=False, onehot=False, protfilter=myfilter)


###

Protein



