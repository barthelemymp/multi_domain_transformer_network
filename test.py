import networkx as nx
import pandas as pd
from itertools import *



### Load the Uniref Graphs
G = nx.read_gml("/Data/PFAM/fullUniref90.gml")

file_to_read = open("unirefname.pkl", "rb")
uniref_dict = pickle.load(file_to_read)
file_to_read = open("unirefarchitecture.pkl", "rb")
uniref_archi = pickle.load(file_to_read)


