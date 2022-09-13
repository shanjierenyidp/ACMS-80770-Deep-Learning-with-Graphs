import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import bipartite
from networkx.generators.random_graphs import erdos_renyi_graph
import copy

# -- Initialize graphs
seed = 30
G = nx.florentine_families_graph()
nodes = G.nodes()
edges = G.edges()
layout = nx.spring_layout(G, seed=seed)

# extracting adjacency matrix
A = nx.adjacency_matrix(G)

# calculate the Jaccard similarity between every pair of nodes 
S = np.zeros((len(nodes),len(nodes)))

for i in range(len(nodes)):
    for j in range(len(nodes)):
        NinNj = np.array(A.todense()[:,i])*np.array(A.todense()[:,j]) # calculate the intersection of N(vi) and N(vj)
        NiuNj = np.array(A.todense()[:,i])+np.array(A.todense()[:,j]) # calculate the union of N(vi) and N(vj)
        S[i,j] = len(np.nonzero(NinNj)[0])/len(np.nonzero(NiuNj)[0]) # put into S matrix
        
# finding the index of the name "Ginori"
Ginori_id = list(nodes).index('Ginori')

# creating the edge dictionary for Similarity to node "Ginori"
S_dict = {}
for i in range(len(nodes)):
    S_dict[list(nodes)[i]] = S[Ginori_id, i]    

# pass the label to the graph.
nx.set_node_attributes(G,S_dict , "Jaccard")

# creating a copy of original graph 
G2 = nx.create_empty_copy(G, with_data=True)
old_edges = copy.deepcopy(G.edges())
new_edges, metric = [], []

# set new edges according to Similarity to node "Ginori"
for i in range(len(G2.nodes)):
    G2.add_edge('Ginori', list(nodes)[i])
    # print(f"({u}, {v}) -> {p:.8f}")
    new_edges.append(('Ginori', list(nodes)[i]))
    metric.append(S_dict[list(nodes)[i]])


# -- plot Florentine Families graph
nx.draw_networkx_nodes(G2, nodelist=nodes, pos=layout, node_size=600)
# -- plot edges representing similarity
import matplotlib as mpl


cmap = mpl.cm.rainbow
norm = mpl.colors.Normalize(vmin=5, vmax=10)


ne = nx.draw_networkx_edges(G2, edgelist=new_edges, pos=layout, edge_color =[ele for ele in S_dict.values()], width=4, edge_cmap = cmap)

"""
    This example is randomly plotting similarities between 8 pairs of nodes in the graph. 
    Identify the ”Ginori”
"""

plt.colorbar(ne)
plt.axis('off')
# plt.show()
plt.savefig('./Jaccard.png')
