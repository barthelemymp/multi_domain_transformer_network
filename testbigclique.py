import networkx as nx
import pandas as pd
import numpy as np
from itertools import *
from src.ProteinTransformer import *
from src.ProteinsDataset import *
from src.MatchingLoss import *
import pickle

### Load the Uniref dict
"""This dict send True if the protein is in uniref and False otherwise (take the uniprot name as keys)"""
def def_value():
    return False
file_to_read = open("/Data/PFAM/unirefname.pkl", "rb")
uniref_dict = pickle.load(file_to_read)



"""This dict send the list of domains present in the protein"""
def def_valuegraph():
    return []
file_to_read = open("/Data/PFAM/unirefarchitecture.pkl", "rb")
uniref_archi = pickle.load(file_to_read)

""" This fonction aims at creating a mask to filter the sequences that are not in Uniref
 and that are not coaapearing with another domain of the clique. to do I load the msa a first time, it is
 unefficient, I ll make it more sensible after"""
def myfilter(protname):
    mask=[]
    for prot in protname:
        if uniref_dict[prot]==False:
            mask.append(False)
            print("not in uniref")
        elif np.sum([prot in pmsa1.protlist, prot in pmsa2.protlist, prot in pmsa3.protlist, prot in pmsa4.protlist, prot in pmsa5.protlist])<2:
            mask.append(False)
            print("alone")
        else:
            mask.append(True)
            print("ok")
    return np.array(mask)
pmsa1 = ProteinMSA("/Data/PFAM/MSAs/PF01799", onehot=False, protfilter =None)
pmsa2 = ProteinMSA("/Data/PFAM/MSAs/PF00111", onehot=False, protfilter =None)
pmsa3 = ProteinMSA("/Data/PFAM/MSAs/PF03450", onehot=False, protfilter =None)
pmsa4 = ProteinMSA("/Data/PFAM/MSAs/PF00941", onehot=False, protfilter =None)
pmsa5 = ProteinMSA("/Data/PFAM/MSAs/PF02738", onehot=False, protfilter =None)



pathClique_list = ["/Data/PFAM/MSAs/PF01799", "/Data/PFAM/MSAs/PF00111", "/Data/PFAM/MSAs/PF03450", "/Data/PFAM/MSAs/PF00941", "/Data/PFAM/MSAs/PF02738"]#, "/Data/PFAM/MSAs/PF12796"]#, "/Data/PFAM/MSAs/PF00069"]#, "PF12796_rp35.txt"]
NameClique_list= ["PF01799", "PF00111", "PF03450", "PF00941", "PF02738"]

# np.sum([prot in pmsa1.protlist, prot in pmsa2.protlist, prot in pmsa3.protlist])
# NameClique_list_as_set = set(NameClique_list)
# intersection = len(NameClique_list_as_set.intersection(uniref_archi['A0A075A3D5']))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pnd = ProteinNetworkDataset( pathClique_list, NameClique_list,  mapstring="-ACDEFGHIKLMNPQRSTVWY", transform=None, device=device, batch_first=False, returnIndex=False, onehot=False, protfilter=myfilter)
dl = DataLoader(pnd, batch_size=50,
                    shuffle=True, num_workers=0, collate_fn=network_collate)




# importlib.reload(ptr)
# importlib.reload(prd)
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
forward_expansion = 124
src_vocab_size = 22
trg_vocab_size = 22
embedding_size = 55

onehot=False
batch_size = 32
num_heads = 1
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
forward_expansion = 2048
src_vocab_size = 22
trg_vocab_size = 22
embedding_size = 55
lr = 0.00001
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


cn = ContextNetwork(translist, pnd.NameClique_list, 50)


params = []
for trans in cn.transformer_list:
    params +=list(trans.parameters())

optimizer = optim.AdamW(params, lr=0.0001, weight_decay=0)

padIndex =pnd.clique[pnd.NameClique_list[0]].SymbolMap["<pad>"]
onehot=False
criterion = nn.CrossEntropyLoss(ignore_index=padIndex)
criterion_raw = nn.CrossEntropyLoss(ignore_index=padIndex, reduction='none')
num_epochs =100
for epoch in range(num_epochs+1):
    print(f"[Epoch {epoch} / {num_epochs}]")
    cn.train()
    lossesCE = []
    for batch_idx, batch in enumerate(dl):
        optimizer.zero_grad()
        memory = cn.encode(batch)
        reconstruct, sizeTable, memorymask_list = cn.reconstruct_encoding(memory)
        output = cn.decodeAll(batch, reconstruct, sizeTable, memorymask_list)
        output = [out for out in output if out !=None]
        loss = NetworkLoss(output, criterion, onehot)
        lossesCE.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 
        optimizer.step()
    mean_lossCETrain = sum(lossesCE) / len(lossesCE)
    print(mean_lossCETrain)

