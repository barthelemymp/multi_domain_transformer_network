
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import math
import numpy as np
import pandas as pd
# from utils import *
from torch._six import string_classes
import collections
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


"""
    >>> from scipy.cluster import hierarchy
    >>> rng = np.random.default_rng()
    >>> X = rng.standard_normal((10, 10))
    >>> Z = hierarchy.ward(X)
    >>> hierarchy.leaves_list(Z)
    array([0, 3, 1, 9, 2, 5, 7, 4, 6, 8], dtype=int32)
    >>> hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, X))
    array([3, 0, 2, 5, 7, 4, 8, 6, 9, 1], dtype=int32)
"""


def read_fasta(fasta_path, alphabet='ACDEFGHIKLMNPQRSTVWY-', default_index=20, filterDic=None):

    # read all the sequences into a dictionary
    seq_dict = {}
    with open(fasta_path, 'r') as file_handle:
        seq_id = None
        for line in file_handle:
            line = line.strip()
            if line.startswith(">"):
                seq_id = line
                seq_dict[seq_id] = ""
                continue
            assert seq_id is not None
            line = ''.join([c for c in line if c.isupper() or c == '-'])
            seq_dict[seq_id] += line

    aa_index = defaultdict(lambda: default_index, {
                           alphabet[i]: i for i in range(len(alphabet))})

    seq_msa = []
    keys_list = []
    for k in seq_dict.keys():
        prot = k.split(">")[-1].split("_")[0]
        if filterDic:
            if not filterDic[prot]:
                continue
        seq_msa.append([aa_index[s] for s in seq_dict[k]])
        keys_list.append(k)

    seq_msa = np.array(seq_msa, dtype=int)

    # # reweighting sequences
    # seq_weight = np.zeros(seq_msa.shape)
    # for j in range(seq_msa.shape[1]):
    #     aa_type, aa_counts = np.unique(seq_msa[:, j], return_counts=True)
    #     num_type = len(aa_type)
    #     aa_dict = {}
    #     for a in aa_type:
    #         aa_dict[a] = aa_counts[list(aa_type).index(a)]
    #     for i in range(seq_msa.shape[0]):
    #         seq_weight[i, j] = (1.0 / num_type) * (1.0 / aa_dict[seq_msa[i, j]])
    # tot_weight = np.sum(seq_weight)
    # seq_weight = seq_weight.sum(1) / tot_weight

    return seq_msa, keys_list, len(alphabet)


def perdomain_collate(batch):
    print("perdombatc",batch, type(batch))
    r"""Puts each data field into a tensor with first dimension batch size. 
    Modified to get 1st dim as batch dim"""

    ordermemory = torch.tensor(np.where([x!=None for x in batch])[0])
    
    print("tensor")
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(batch[i].numel() for i in ordermemory)
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage)
    print(ordermemory)
    #print([batch[i] for i in ordermemory])
    return torch.stack([batch[i] for i in ordermemory], 1, out=out), ordermemory




def network_collate(batch):
    print(batch, type(batch))


    print("Sequence")
    # check to make sure that the elements in batch have consistent size
    it = iter(batch)
    elem_size = len(next(it))
    if not all(len(elem) == elem_size for elem in it):
        raise RuntimeError(
            'each element in list of batch should be of equal size')
    transposed = zip(*batch)
    return [perdomain_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

















class ProteinMSA(torch.utils.data.Dataset):
    def __init__(self, fastaPath,  mapstring='ACDEFGHIKLMNPQRSTVWY-', transform=None, device=None, onehot=True, batch_first=False):
        """
        Args:
            fastaPath (string): Path to the fasta file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        seq_nat, ks, q = read_fasta(fastaPath)
        self.onehot = onehot
        self.q = q
        # self.get_fitness = get_fitness
        # if get_fitness != None:
        #     self.fitness = torch.tensor(list(map(get_fitness, ks)))
        def getprotname(k):
            return k.split(">")[-1].split("_")[0]
        self.protlist = list(map(getprotname, ks))
        self.prot2pos = defaultdict(lambda x:None, {self.protlist[i]:i for i in range(len(self.protlist))})
        self.mapstring = mapstring
        self.SymbolMap = dict([(mapstring[i], i)
                              for i in range(len(mapstring))])
        
        self.SymbolMap["<pad>"] = self.q+1
        self.q += 1


        self.nseq, self.len_protein = seq_nat.shape
        self.padsequence = torch.tensor([self.q]*self.len_protein).to(self.device)
#         seq_msa = torch.from_numpy(seq_msa)
#         train_msa = one_hot(seq_msa, num_classes=num_res_type).cuda()
#         train_msa = train_msa.view(train_msa.shape[0], -1).float()

        # self.train_weight = torch.from_numpy(w_nat)
        # self.train_weight = (self.train_weight / torch.sum(self.train_weight))
        self.gap = "-"

        if onehot:
            train_msa = torch.nn.functional.one_hot(
                torch.from_numpy(seq_nat).long(), num_classes=self.q)
            if flatten:
                self.sequences = train_msa.view(train_msa.shape[0], -1).float()
            else:
                self.sequences = train_msa.float()
        else:
            self.sequences = torch.from_numpy(seq_nat).long()

        self.device = device
        self.transform = transform


        self.batch_first = batch_first

        if not batch_first:
            self.sequences = torch.transpose(self.sequences, 0,1)


        if device != None:
            # self.train_weight = self.train_weight.to(device, non_blocking=True)
            self.sequences = self.sequences.to(device, non_blocking=True)
            # if get_fitness != None:
            #     self.fitness = self.fitness.to(device, non_blocking=True)

    def __len__(self):
        return self.sequences.shape[0]
#         if self.batch_first:
#         else:
#             return self.tensorIN.shape[1]

    # from the dataset, gives the data in the form it will be used by the NN
    def __getitem__(self, idx):
        print(idx)
        if torch.is_tensor(idx):
            
            idx = idx.tolist()
            print(idx)
#             if len(idx) > 1:
#                 raise Exception("getitem recieved more than one value... change code")
        if idx<0:
            return None
        # if self.get_fitness != None:
        #     return self.sequences[idx, :], self.train_weight[idx], self.fitness[idx]
        # else:
        #     return self.sequences[:, idx]#, self.train_weight[idx]
        if self.batch_first:
            return self.sequences[idx]
        else:
            return self.sequences[:,idx]

    # def uniformWeights(self):
    #     self.train_weight = torch.ones(self.train_weight.shape)

    def to_(self, device):
        if device != None:
            # self.train_weight = self.train_weight.to(device, non_blocking=True)
            self.sequences = self.sequences.to(device, non_blocking=True)
            # if self.get_fitness != None:
            #     self.fitness = self.fitness.to(device, non_blocking=True)

    def pad(self, maxlen, padsymbol="<pad>"):
        # self.SymbolMap["<pad>"] = self.q+1
        # self.q += 1
        # TO DO

    def terminate(self, sos="<sos>", eos="<eos>"):
        self.init_token = sos
        self.eos_token = eos
        if sos not in self.SymbolMap:
            self.SymbolMap[sos] = self.q
            self.q += 1

        if eos not in self.SymbolMap:
            self.SymbolMap[eos] = self.q
            self.q += 1

        if self.onehot:
            seq = self.sequences.max(dim=2)[1].float()
            eos_list = torch.ones(self.sequences.shape[0], 1).to(
                self.sequences.device)*self.SymbolMap[eos]
            sos_list = torch.ones(self.sequences.shape[0], 1).to(
                self.sequences.device)*self.SymbolMap[sos]
            if self.batch_first:
                seq = torch.cat([sos_list, seq, eos_list], dim=1)
            else:
                seq = torch.cat([sos_list, seq, eos_list], dim=0)
            self.sequences = torch.nn.functional.one_hot(
                seq.long(), num_classes=self.q)

        else:
            eos_list = torch.ones(self.sequences.shape[0], 1).to(
                self.sequences.device)*self.SymbolMap[eos]
            sos_list = torch.ones(self.sequences.shape[0], 1).to(
                self.sequences.device)*self.SymbolMap[sos]
            if self.batch_first:
                self.sequences = torch.cat(
                    [sos_list, self.sequences, eos_list], dim=1)
            else:
                self.sequences = torch.cat(
                    [sos_list, self.sequences, eos_list], dim=0)
        self.len_protein += 2


class ProteinNetworkDataset(torch.utils.data.Dataset):
    def __init__(self, pathClique_list, NameClique_list,  mapstring="-ACDEFGHIKLMNPQRSTVWY", transform=None, device=None, batch_first=False, returnIndex=False, onehot=True):

        
        self.NameClique_list = NameClique_list
        self.q = len(mapstring)
        self.SymbolMap = dict([(mapstring[i], i)
                              for i in range(len(mapstring))])
        self.init_token = "<sos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.unk = "X"

        self.mapstring = mapstring
        self.onehot = onehot

        self.SymbolMap[self.unk] = len(mapstring)
        self.gap = "-"

        self.device = device
        self.transform = transform
        self.batch_first = batch_first
        self.returnIndex = returnIndex
        
        self.clique = {}
        unique = []
        for path, name in zip(pathClique_list, NameClique_list):
            self.clique[name] = ProteinMSA(path,  mapstring=mapstring, transform=transform, device=device, onehot=onehot, batch_first=batch_first)
            unique +=self.clique[name].prot2pos.keys()
        self.uniqueProt = np.unique(unique)
        self.nDom = len(NameClique_list)
        self.idx_in_clique = torch.zeros((len(self.uniqueProt), self.nDom ), dtype=torch.long, device=self.device)
        
        for i in range(len(self.uniqueProt)):
            prot = self.uniqueProt[i]
            for j in range(self.nDom):
                fam = NameClique_list[j]
                if prot in self.clique[fam].prot2pos.keys():
                    self.idx_in_clique[i,j] = self.clique[fam].prot2pos[prot]
                else :
                    self.idx_in_clique[i,j] = -1
                    
        
    
    def __len__(self):
        return len(self.uniqueProt)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return [self.clique[fam][self.idx_in_clique[idx,j]] for j,fam in enumerate(self.NameClique_list)]


    # def to(self, device):
    #     if device != None:
    #         self.tensorIN = self.tensorIN.to(device, non_blocking=True)
    #         self.tensorOUT = self.tensorOUT.to(device, non_blocking=True)

    # def shufflePairs(self,):
    #     self.tensorOUT = self.tensorOUT[:,
    #                                     torch.randperm(self.tensorOUT.size()[1])]

    # def downsample(self, nsamples):
    #     idxs = torch.randperm(self.tensorOUT.size()[1])[:nsamples]
    #     self.tensorIN = self.tensorIN[:, idxs]
    #     self.tensorOUT = self.tensorOUT[:, idxs]

    # def join(self, pds):
    #     if self.device != pds.device:
    #         pds.tensorIN= pds.tensorIN.to(self.device, non_blocking=True)
    #         pds.tensorOUT= pds.tensorOUT.to(self.device, non_blocking=True)
    #     if self.onehot:
    #         if self.inputsize < pds.inputsize:
    #             dif = pds.inputsize - self.inputsize
    #             padIN = torch.zeros(dif, len(self), len(self.SymbolMap)).to(self.device, non_blocking=True)
    #             for i in range(len(self)):
    #                 inp = [self.SymbolMap[self.pad_token]]*dif
    #                 padIN[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
    #             self.tensorIN = torch.cat([torch.cat([self.tensorIN, padIN],dim=0), pds.tensorIN], dim=1)
    #             self.inputsize = pds.inputsize
    #         elif self.inputsize > pds.inputsize:
    #             dif = self.inputsize - pds.inputsize
    #             padIN = torch.zeros(dif, len(pds), len(self.SymbolMap)).to(self.device, non_blocking=True)
    #             for i in range(len(pds)):
    #                 inp = [self.SymbolMap[self.pad_token]] * dif
    #                 padIN[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
    #             self.tensorIN = torch.cat([self.tensorIN, torch.cat([pds.tensorIN, padIN],dim=0)], dim=1)
    #             pds.inputsize = self.inputsize
    #         if self.outputsize < pds.outputsize:
    #             dif = pds.outputsize - self.outputsize
    #             padOUT = torch.zeros(dif, self.tensorOUT.shape[1], len(self.SymbolMap)).to(self.device, non_blocking=True)
    #             for i in range(self.tensorOUT.shape[1]):
    #                 inp = [self.SymbolMap[self.pad_token]]*dif
    #                 padOUT[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
    #             self.tensorOUT = torch.cat([torch.cat([self.tensorOUT, padOUT],dim=0), pds.tensorOUT], dim=1)
    #             self.outputsize = pds.outputsize
    #         elif self.outputsize > pds.outputsize:
    #             dif = self.outputsize - pds.outputsize
    #             padOUT = torch.zeros(dif, pds.tensorOUT.shape[1], len(self.SymbolMap)).to(self.device, non_blocking=True)
    #             for i in range(pds.tensorOUT.shape[1]):
    #                 inp = [self.SymbolMap[self.pad_token]] * dif
    #                 padOUT[:,i,:] = torch.nn.functional.one_hot(torch.tensor(inp), num_classes=len(self.SymbolMap))
    #             self.tensorOUT = torch.cat([self.tensorOUT, torch.cat([pds.tensorOUT, padOUT],dim=0)], dim=1)
    #             pds.outputsize = self.outputsize
    #     else:
    #         if self.inputsize < pds.inputsize:
    #             dif = pds.inputsize - self.inputsize
    #             padIN = torch.zeros(dif, len(self)).to(self.device, non_blocking=True)
    #             for i in range(len(self)):
    #                 inp = [self.SymbolMap[self.pad_token]]*dif
    #                 padIN[:,i] = torch.tensor(inp)
    #             self.tensorIN = torch.cat([torch.cat([self.tensorIN, padIN],dim=0), pds.tensorIN], dim=1)
    #             self.inputsize = pds.inputsize
    #         elif self.inputsize > pds.inputsize:
    #             dif = self.inputsize - pds.inputsize
    #             padIN = torch.zeros(dif, len(pds)).to(self.device, non_blocking=True)
    #             for i in range(len(pds)):
    #                 inp = [self.SymbolMap[self.pad_token]] * dif
    #                 padIN[:,i] = torch.tensor(inp)
    #             self.tensorIN = torch.cat([self.tensorIN, torch.cat([pds.tensorIN, padIN],dim=0)], dim=1)
    #             pds.inputsize = self.inputsize
    #         if self.outputsize < pds.outputsize:
    #             dif = pds.outputsize - self.outputsize
    #             padOUT = torch.zeros(dif, self.tensorOUT.shape[1]).to(self.device, non_blocking=True)
    #             for i in range(self.tensorOUT.shape[1]):
    #                 inp = [self.SymbolMap[self.pad_token]]*dif
    #                 padOUT[:,i] = torch.tensor(inp)
    #             self.tensorOUT = torch.cat([torch.cat([self.tensorOUT, padOUT],dim=0), pds.tensorOUT], dim=1)
    #             self.outputsize = pds.outputsize
    #         elif self.outputsize > pds.outputsize:
    #             dif = self.outputsize - pds.outputsize
    #             padOUT = torch.zeros(dif, pds.tensorOUT.shape[1]).to(self.device, non_blocking=True)
    #             for i in range(pds.tensorOUT.shape[1]):
    #                 inp = [self.SymbolMap[self.pad_token]] * dif
    #                 padOUT[:,i] = torch.tensor(inp)
    #             self.tensorOUT = torch.cat([self.tensorOUT, torch.cat([pds.tensorOUT, padOUT],dim=0)], dim=1)
    #             pds.outputsize = self.outputsize


class BucketIterator(Iterator):
    """Defines an iterator that batches examples of similar lengths together.
    Minimizes amount of padding needed while producing freshly shuffled
    batches for each new epoch. See pool for the bucketing procedure used.
    """

    def create_batches(self):
        if self.sort:
            self.batches = batch(self.data(), self.batch_size,
                                 self.batch_size_fn)
        else:
            self.batches = pool(self.data(), self.batch_size,
                                self.sort_key, self.batch_size_fn,
                                random_shuffler=self.random_shuffler,
                                shuffle=self.shuffle,
                                sort_within_batch=self.sort_within_batch)


def batch(data, batch_size, batch_size_fn=None):
    """Yield elements from data in chunks of batch_size."""
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
    if minibatch:
        yield minibatch


def pool(data, batch_size, key, batch_size_fn=lambda new, count, sofar: count,
         random_shuffler=None, shuffle=False, sort_within_batch=False):
    """Sort within buckets, then batch, then shuffle batches.
    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    if random_shuffler is None:
        random_shuffler = random.shuffle
    for p in batch(data, batch_size * 100, batch_size_fn):
        p_batch = batch(sorted(p, key=key), batch_size, batch_size_fn) \
            if sort_within_batch \
            else batch(p, batch_size, batch_size_fn)
        if shuffle:
            for b in random_shuffler(list(p_batch)):
                yield b
        else:
            for b in list(p_batch):
                yield b


def default_collate(batch):
    r"""Puts each data field into a tensor with first dimension batch size. 
    Modified to get 1st dim as batch dim"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 1, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def getUnique(tensor):
    inverseMapping = torch.unique(tensor, dim=1, return_inverse=True)[1]
    dic = defaultdict(lambda: 0)
    BooleanKept = torch.tensor([False] * tensor.shape[1])
    for i in range(tensor.shape[1]):
        da = int(inverseMapping[i])
        if dic[da] == 0:
            BooleanKept[i] = True
        dic[da] += 1
    return tensor[:, BooleanKept, :], BooleanKept


class ProteinTranslationDataset(torch.utils.data.Dataset):
    def __init__(self, csvPath,  mapstring="-ACDEFGHIKLMNPQRSTVWY", transform=None, device=None, batch_first=False, Unalign=False, filteringOption='none', returnIndex=False, onehot=True, GapTheExtra=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df = pd.read_csv(csvPath, header=None)
        self.q = len(mapstring)
        self.SymbolMap = dict([(mapstring[i], i)
                              for i in range(len(mapstring))])
        self.init_token = "<sos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.unk = "X"
        self.GapTheExtra = GapTheExtra
        self.padIndex = -100
        if GapTheExtra:
            self.init_token = "-"
            self.eos_token = "-"
            self.pad_token = "-"
            self.unk = "-"
        self.mapstring = mapstring
        self.onehot = onehot
        self.SymbolMap = dict([(mapstring[i], i)
                              for i in range(len(mapstring))])
        self.SymbolMap[self.unk] = len(mapstring)
        if GapTheExtra == False:
            self.SymbolMap[self.init_token] = len(mapstring)+1
            self.SymbolMap[self.eos_token] = len(mapstring)+2
            self.SymbolMap[self.pad_token] = len(mapstring)+3
            self.padIndex = len(mapstring)+3
        else:
            self.SymbolMap[self.init_token] = 0
            self.SymbolMap[self.eos_token] = 0
            self.SymbolMap[self.pad_token] = 0

        self.inputsize = len(df.iloc[1][0].split(" "))+2
        self.outputsize = len(df.iloc[1][1].split(" "))+2
        self.gap = "-"
        self.tensorIN = torch.zeros(
            self.inputsize, len(df), len(self.SymbolMap))
        self.tensorOUT = torch.zeros(
            self.outputsize, len(df), len(self.SymbolMap))
        self.device = device
        self.transform = transform
        self.batch_first = batch_first
        self.filteringOption = filteringOption
        self.returnIndex = returnIndex
        if Unalign == False:
            print("keeping the gap")
            for i in range(len(df)):
                inp = [self.SymbolMap[self.init_token]]+[self.SymbolMap[k]
                                                         for k in df[0][i].split(" ")]+[self.SymbolMap[self.eos_token]]
                out = [self.SymbolMap[self.init_token]]+[self.SymbolMap[k]
                                                         for k in df[1][i].split(" ")]+[self.SymbolMap[self.eos_token]]
                self.tensorIN[:, i, :] = torch.nn.functional.one_hot(
                    torch.tensor(inp), num_classes=len(self.SymbolMap))
                self.tensorOUT[:, i, :] = torch.nn.functional.one_hot(
                    torch.tensor(out), num_classes=len(self.SymbolMap))
        else:
            print("Unaligning and Padding")
            for i in range(len(df)):
                inp = [self.SymbolMap[self.init_token]]+[self.SymbolMap[k]
                                                         for k in df[0][i].split(" ") if k != self.gap]+[self.SymbolMap[self.eos_token]]
                out = [self.SymbolMap[self.init_token]]+[self.SymbolMap[k]
                                                         for k in df[1][i].split(" ") if k != self.gap]+[self.SymbolMap[self.eos_token]]
                inp += [self.SymbolMap[self.pad_token]] * \
                    (self.inputsize - len(inp))
                out += [self.SymbolMap[self.pad_token]] * \
                    (self.outputsize - len(out))
                self.tensorIN[:, i, :] = torch.nn.functional.one_hot(
                    torch.tensor(inp), num_classes=len(self.SymbolMap))
                self.tensorOUT[:, i, :] = torch.nn.functional.one_hot(
                    torch.tensor(out), num_classes=len(self.SymbolMap))

        if filteringOption == "in":
            a = getUnique(self.tensorIN)[1]
            self.tensorIN = self.tensorIN[:, a, :]
            self.tensorOUT = self.tensorOUT[:, a, :]
            print("filtering the redundancy of input proteins")
        elif filteringOption == "out":
            b = getUnique(self.tensorOUT)[1]
            self.tensorIN = self.tensorIN[:, b, :]
            self.tensorOUT = self.tensorOUT[:, b, :]
            print("filtering the redundancy of output proteins")
        elif filteringOption == "and":
            a = getUnique(self.tensorIN)[1]
            b = getUnique(self.tensorOUT)[1]
            self.tensorIN = self.tensorIN[:, a*b, :]
            self.tensorOUT = self.tensorOUT[:, a*b, :]
            print("filtering the redundancy of input AND output proteins")
        elif filteringOption == "or":
            a = getUnique(self.tensorIN)[1]
            b = getUnique(self.tensorOUT)[1]
            self.tensorIN = self.tensorIN[:, a+b, :]
            self.tensorOUT = self.tensorOUT[:, a+b, :]
            print("filtering the redundancy of input OR output proteins")
        else:
            print("No filtering of redundancy")

        if batch_first:
            self.tensorIN = torch.transpose(self.tensorIN, 0, 1)
            self.tensorOUT = torch.transpose(self.tensorOUT, 0, 1)

        if onehot == False:
            self.tensorIN = self.tensorIN.max(dim=2)[1]
            self.tensorOUT = self.tensorOUT.max(dim=2)[1]

        if device != None:
            self.tensorIN = self.tensorIN.to(device, non_blocking=True)
            self.tensorOUT = self.tensorOUT.to(device, non_blocking=True)

    def __len__(self):
        if self.batch_first:
            return self.tensorIN.shape[0]
        else:
            return self.tensorIN.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.batch_first:
            if self.returnIndex:
                return self.tensorIN[idx, :], self.tensorOUT[idx, :], idx
            else:
                return self.tensorIN[idx, :], self.tensorOUT[idx, :]
        else:
            if self.returnIndex:
                return self.tensorIN[:, idx], self.tensorOUT[:, idx], idx
            else:
                return self.tensorIN[:, idx], self.tensorOUT[:, idx]

    def to(self, device):
        if device != None:
            self.tensorIN = self.tensorIN.to(device, non_blocking=True)
            self.tensorOUT = self.tensorOUT.to(device, non_blocking=True)

    def shufflePairs(self,):
        self.tensorOUT = self.tensorOUT[:,
                                        torch.randperm(self.tensorOUT.size()[1])]

    def downsample(self, nsamples):
        idxs = torch.randperm(self.tensorOUT.size()[1])[:nsamples]
        self.tensorIN = self.tensorIN[:, idxs]
        self.tensorOUT = self.tensorOUT[:, idxs]

    def join(self, pds):
        if self.device != pds.device:
            pds.tensorIN = pds.tensorIN.to(self.device, non_blocking=True)
            pds.tensorOUT = pds.tensorOUT.to(self.device, non_blocking=True)
        if self.onehot:
            if self.inputsize < pds.inputsize:
                dif = pds.inputsize - self.inputsize
                padIN = torch.zeros(dif, len(self), len(self.SymbolMap)).to(
                    self.device, non_blocking=True)
                for i in range(len(self)):
                    inp = [self.SymbolMap[self.pad_token]]*dif
                    padIN[:, i, :] = torch.nn.functional.one_hot(
                        torch.tensor(inp), num_classes=len(self.SymbolMap))
                self.tensorIN = torch.cat(
                    [torch.cat([self.tensorIN, padIN], dim=0), pds.tensorIN], dim=1)
                self.inputsize = pds.inputsize
            elif self.inputsize > pds.inputsize:
                dif = self.inputsize - pds.inputsize
                padIN = torch.zeros(dif, len(pds), len(self.SymbolMap)).to(
                    self.device, non_blocking=True)
                for i in range(len(pds)):
                    inp = [self.SymbolMap[self.pad_token]] * dif
                    padIN[:, i, :] = torch.nn.functional.one_hot(
                        torch.tensor(inp), num_classes=len(self.SymbolMap))
                self.tensorIN = torch.cat(
                    [self.tensorIN, torch.cat([pds.tensorIN, padIN], dim=0)], dim=1)
                pds.inputsize = self.inputsize
            if self.outputsize < pds.outputsize:
                dif = pds.outputsize - self.outputsize
                padOUT = torch.zeros(dif, self.tensorOUT.shape[1], len(
                    self.SymbolMap)).to(self.device, non_blocking=True)
                for i in range(self.tensorOUT.shape[1]):
                    inp = [self.SymbolMap[self.pad_token]]*dif
                    padOUT[:, i, :] = torch.nn.functional.one_hot(
                        torch.tensor(inp), num_classes=len(self.SymbolMap))
                self.tensorOUT = torch.cat(
                    [torch.cat([self.tensorOUT, padOUT], dim=0), pds.tensorOUT], dim=1)
                self.outputsize = pds.outputsize
            elif self.outputsize > pds.outputsize:
                dif = self.outputsize - pds.outputsize
                padOUT = torch.zeros(dif, pds.tensorOUT.shape[1], len(
                    self.SymbolMap)).to(self.device, non_blocking=True)
                for i in range(pds.tensorOUT.shape[1]):
                    inp = [self.SymbolMap[self.pad_token]] * dif
                    padOUT[:, i, :] = torch.nn.functional.one_hot(
                        torch.tensor(inp), num_classes=len(self.SymbolMap))
                self.tensorOUT = torch.cat(
                    [self.tensorOUT, torch.cat([pds.tensorOUT, padOUT], dim=0)], dim=1)
                pds.outputsize = self.outputsize
        else:
            if self.inputsize < pds.inputsize:
                dif = pds.inputsize - self.inputsize
                padIN = torch.zeros(dif, len(self)).to(
                    self.device, non_blocking=True)
                for i in range(len(self)):
                    inp = [self.SymbolMap[self.pad_token]]*dif
                    padIN[:, i] = torch.tensor(inp)
                self.tensorIN = torch.cat(
                    [torch.cat([self.tensorIN, padIN], dim=0), pds.tensorIN], dim=1)
                self.inputsize = pds.inputsize
            elif self.inputsize > pds.inputsize:
                dif = self.inputsize - pds.inputsize
                padIN = torch.zeros(dif, len(pds)).to(
                    self.device, non_blocking=True)
                for i in range(len(pds)):
                    inp = [self.SymbolMap[self.pad_token]] * dif
                    padIN[:, i] = torch.tensor(inp)
                self.tensorIN = torch.cat(
                    [self.tensorIN, torch.cat([pds.tensorIN, padIN], dim=0)], dim=1)
                pds.inputsize = self.inputsize
            if self.outputsize < pds.outputsize:
                dif = pds.outputsize - self.outputsize
                padOUT = torch.zeros(dif, self.tensorOUT.shape[1]).to(
                    self.device, non_blocking=True)
                for i in range(self.tensorOUT.shape[1]):
                    inp = [self.SymbolMap[self.pad_token]]*dif
                    padOUT[:, i] = torch.tensor(inp)
                self.tensorOUT = torch.cat(
                    [torch.cat([self.tensorOUT, padOUT], dim=0), pds.tensorOUT], dim=1)
                self.outputsize = pds.outputsize
            elif self.outputsize > pds.outputsize:
                dif = self.outputsize - pds.outputsize
                padOUT = torch.zeros(dif, pds.tensorOUT.shape[1]).to(
                    self.device, non_blocking=True)
                for i in range(pds.tensorOUT.shape[1]):
                    inp = [self.SymbolMap[self.pad_token]] * dif
                    padOUT[:, i] = torch.tensor(inp)
                self.tensorOUT = torch.cat(
                    [self.tensorOUT, torch.cat([pds.tensorOUT, padOUT], dim=0)], dim=1)
                pds.outputsize = self.outputsize


def getPreciseBatch(pds, idxToget):
    data = []
    for idx in idxToget:
        data.append(pds[idx])
    batch = default_collate(data)
    return batch


def getBooleanisRedundant(tensor1, tensor2):
    l1 = tensor1.shape[1]
    l2 = tensor2.shape[1]
    BooleanKept = torch.tensor([True]*l2)
    for i in range(l1):
        protein1 = tensor1[:, i, :]
        for j in range(l2):
            protein2 = tensor2[:, j, :]
            if torch.equal(protein1, protein2):
                BooleanKept[j] = False
    return BooleanKept


def deleteRedundancyBetweenDatasets(pds1, pds2):

    filteringOption = pds1.filteringOption
    if filteringOption == "in":
        a = getBooleanisRedundant(pds1.tensorIN, pds2.tensorIN)
        pds2.tensorIN = pds2.tensorIN[:, a, :]
        pds2.tensorOUT = pds2.tensorOUT[:, a, :]
    elif filteringOption == "out":
        b = getBooleanisRedundant(pds1.tensorOUT, pds2.tensorOUT)
        pds2.tensorIN = pds2.tensorIN[:, b, :]
        pds2.tensorOUT = pds2.tensorOUT[:, b, :]
    elif filteringOption == "and":
        a = getBooleanisRedundant(pds1.tensorIN, pds2.tensorIN)
        b = getBooleanisRedundant(pds1.tensorOUT, pds2.tensorOUT)
        pds2.tensorIN = pds2.tensorIN[:, a*b, :]
        pds2.tensorOUT = pds2.tensorOUT[:, a*b, :]
    elif filteringOption == "or":
        a = getBooleanisRedundant(pds1.tensorIN, pds2.tensorIN)
        b = getBooleanisRedundant(pds1.tensorOUT, pds2.tensorOUT)
        pds2.tensorIN = pds2.tensorIN[:, a+b, :]
        pds2.tensorOUT = pds2.tensorOUT[:, a+b, :]
