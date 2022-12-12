"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 2: Programming assignment
Problem 2
"""
import torch
import torch
from torch import nn
import warnings
import numpy as np
from torch.nn.parameter import Parameter
import mydataset
from torch.nn.modules.module import Module
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian


torch.manual_seed(0)


class GCN(Module):
    """
        Graph convolutional layer
    """
    def __init__(self, in_features, out_features, bias = False):
        super(GCN,self).__init__()
        # -- initialize weight
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # -- non-linearity
        self.nonlinear = torch.nn.functional.relu
        
        torch.nn.init.xavier_normal_(self.weight)
    def __call__(self, A, H):
        # print(A,H)
        if len(H.size())==2:
            H = torch.unsqueeze(H,dim=0)
        if len(A.size())==2:
            A = torch.unsqueeze(A,dim=0)
        A_tilde = A+torch.eye(A.size()[1])
        # print(A_tilde)
        # -- GCN propagation rule
        D1_2 = []
        for i in range(len(H)):
            D_temp = torch.diag(torch.pow(torch.sum(A_tilde[i], axis=0),-0.5))
            D1_2.append(D_temp)
        self.D1_2 = torch.stack(D1_2)
        # print(self.D1_2)
        output = torch.matmul(self.D1_2,A_tilde)
        # print(output)
        output = torch.matmul(output,self.D1_2)
        # print(output)
        output = torch.matmul(output, H)
        # print(output)
        output = torch.matmul(output,self.weight)
        # print(output)
        # output = torch.chain_matmul(torch.pow(self.D,-0.5),A_tilde,torch.pow(self.D,-0.5),H,self.weight)
        
        
        return self.nonlinear(output).squeeze()


class MyModel(nn.Module):
    """
        Regression  model
    """
    def __init__(self, layer_dimensions):
        super(MyModel, self).__init__()
        # -- initialize layers
        self.layer_dimensions = layer_dimensions
        # self.output_dimensions = output_dimension
        self.convs = []
        for i in range(len(layer_dimensions)-1):
            self.convs.append(GCN(layer_dimensions[i],layer_dimensions[i+1] ))

    def forward(self, A, h0):
        # print(A,h0)
        # print('first output',output.size())
        output = h0
        for i in range(len(self.convs)):
            output = self.convs[i](A,output)
            # print(i+1,output)
        return output


"""
    Effective range
"""
# -- Initialize graph
seed = 32
n_V = 200   # total number of nodes
i = 17      # node ID  17, 27 
k = 6      # k-hop  2,4,6
G = nx.barabasi_albert_graph(n_V, 2, seed=seed)
A = torch.tensor(nx.adjacency_matrix(G).todense())

# import sys
# sys.exit("some error message")

# -- plot graph
layout = nx.spring_layout(G, seed=seed, iterations=400)
f1 = plt.figure()
# nx.draw(g, ax=f.add_subplot(111))
nx.draw(G, pos=layout, edge_color='gray', width=2, with_labels=False, node_size=50)
f1.savefig("./results/graph.png")

# -- plot neighborhood
nodes = nx.single_source_shortest_path_length(G, i, cutoff=k)
f2 = plt.figure()

im2 = nx.draw_networkx(G, nodelist=nodes, edge_color='gray', font_color = 'k',font_size = 10,font_weight = 'bold',  labels = {i:str(i)}, pos=layout, node_color='red', node_size=100)

# -- visualize
f2.colorbar(im2)
f2.savefig('./results/nodes_i{:d}_k{:d}.png'.format(i,k))


"""
    Influence score
"""
# -- Initialize the model and node feature vectors
# model =  MyModel([200, 100]); model_id = 1
# model =  MyModel([200, 100, 50, 20]); model_id = 2
model =  MyModel([200, 100, 50, 20, 20, 20]); model_id = 3

H = torch.eye(n_V, requires_grad =True) # initialize with one-hot-vector for node indicies 
# Hk = torch.sum(H)*torch.rand(200,100)
Hk = model(A,H)
# print(Hk)
# create seed vector for node i
id_i = 27 # 17, 27


seed_vector = torch.zeros_like(Hk)
seed_vector[id_i] = 1
Hk.backward(seed_vector)
gradient = H.grad

# -- Influence sore
for id_j in range(n_V):
    extract_vector = torch.zeros(n_V).unsqueeze(0)
    extract_vector[0,id_j] = 1
    temp_score = torch.sum(torch.matmul(extract_vector, H.grad))
    # add inf_score to the graph
    G.add_node(id_j,inf_score=temp_score)

# print(inf_score)
# inf_score = ?

# -- plot influence scores
# print(G.nodes())
# print(G.edges())
# print(G.nodes[inf_score])

# for k,v in G.nodes(data=True):
    # print(k,v.keys(), v.values())
# print(G.nodes)

# print([G.nodes[i]['inf_score'].numpy() for i in range(n_V)])
inf_scores = [abs(G.nodes[i]['inf_score'].numpy()) for i in range(n_V)]
inf_score_min, inf_score_max = min(inf_scores),max(inf_scores)
print('min',inf_score_min, 'max',inf_score_max)
# print(G.nodes.data(inf_score))
# f3 = plt.figure()
#figure single
import matplotlib.pylab as pylab
params = {'legend.fontsize': 25,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
pylab.rcParams.update(params)
f3, ax= plt.subplots(figsize = (12,10))
#vmin=0, vmax=1 ,
im3 = nx.draw_networkx(G,ax =ax, vmin=0, vmax=inf_score_max ,labels = {id_i:str(id_i)},  edge_color='gray', with_labels = True,font_color = 'k',font_size = 15,font_weight = 'bold', pos=layout, node_size=100 , node_color=inf_scores, cmap = 'rainbow') # 
# f3.colorbar(im3, cmap='rainbow')#,boundaries = [0,inf_score_max], ticks = [0, inf_score_max])
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
cmap = mpl.cm.rainbow
norm = mpl.colors.Normalize(vmin=0,vmax=inf_score_max)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
import math
plt.colorbar(sm, ticks=[math.floor(ele*1000)/1000 for ele in np.linspace(0,inf_score_max,4)])
            #  boundaries=np.arange(-0.05,2.1,.1))
# # -- visualize
f3.savefig('./results/nodes_inf_score_Mid{:d}_Vid{:d}.png'.format(model_id, id_i))
