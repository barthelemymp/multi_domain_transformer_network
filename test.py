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
            


pmsa1 = ProteinMSA("/Data/PFAM/MSAs/PF13857", onehot=False, protfilter =myfilter)
pmsa2 = ProteinMSA("/Data/PFAM/MSAs/PF00023", onehot=False, protfilter =myfilter)
pmsa3 = ProteinMSA("/Data/PFAM/MSAs/PF12796", onehot=False, protfilter =myfilter)
pmsa4 = ProteinMSA("/Data/PFAM/MSAs/PF00069", onehot=False, protfilter =myfilter)


pathClique_list = ["/Data/PFAM/MSAs/PF13857", "/Data/PFAM/MSAs/PF00023", "/Data/PFAM/MSAs/PF12796", "/Data/PFAM/MSAs/PF00069"]#, "PF12796_rp35.txt"]
NameClique_list= ["PF13857", "PF00023", "PF12796", "PF00069"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pnd = ProteinNetworkDataset( pathClique_list, NameClique_list,  mapstring="-ACDEFGHIKLMNPQRSTVWY", transform=None, device=device, batch_first=False, returnIndex=False, onehot=False, protfilter=myfilter)
dl = DataLoader(pnd, batch_size=50,
                    shuffle=True, num_workers=0, collate_fn=network_collate)




### Transformer Network
torch.set_num_threads(8)
#pathtoFolder = "/home/Datasets/DomainsInter/processed/"
count = 0
torch.set_num_threads(3)
# Model hyperparameters--> CAN BE CHANGED
batch_size = 32
num_heads = 1
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
forward_expansion = 2048
src_vocab_size = 22
trg_vocab_size = 22
embedding_size = 55


batch_size = 32
num_heads = 1
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
forward_expansion = 2048
src_vocab_size = 22
trg_vocab_size = 22
embedding_size = 55

translist = []
for i in range(len(NameClique_list)):
    src_pad_idx = pnd.clique[pnd.NameClique_list[i]].SymbolMap["<pad>"]
    pnd.clique[pnd.NameClique_list[i]] 

    src_position_embedding = PositionalEncoding(embedding_size, max_len=pnd.clique[pnd.NameClique_list[i]].len_protein,device=device)
    trg_position_embedding = PositionalEncoding(embedding_size, max_len=pnd.clique[pnd.NameClique_list[i]].len_protein, device=device)


    model = Transformer(
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        src_position_embedding,
        trg_position_embedding,
        device,
        onehot=onehot,
    ).to(device)
    translist.append(model)


cn = ContextNetwork(translist, pnd.NameClique_list, 5)
