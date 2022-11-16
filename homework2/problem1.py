"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 2: Programming assignment
Problem 1
"""
import torch
from torch import nn
import warnings
import numpy as np
from torch.nn.parameter import Parameter
warnings.simplefilter(action='ignore', category=UserWarning)
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor
import mydataset
from torch.nn.modules.module import Module
from tqdm import tqdm
"""
    load data
"""
print('loading data')
dataset, dataset_smiles = mydataset.get_qm9(GGNNPreprocessor(kekulize=True), return_smiles=True,
                                            target_index=np.random.choice(range(133000), 6000, False))

                                           

V = 9 # this is the number of nodes, they made it uniform
atom_types = [6, 8, 7, 9, 1]


def adj(x):
    x = x[1]
    adjacency = np.zeros((V, V)).astype(float)
    adjacency[:len(x[0]), :len(x[0])] = x[0] + 2 * x[1] + 3 * x[2]
    return torch.FloatTensor(adjacency)


def sig(x):
    x = x[0]
    atoms = np.ones((V)).astype(float)
    atoms[:len(x)] = x
    out = np.array([int(atom == atom_type) for atom_type in atom_types for atom in atoms]).astype(float)
    return torch.FloatTensor(out).reshape(5, len(atoms)).T


def target(x):
    x = x[2] # i assume this is the HOMO property then 
    return torch.FloatTensor(x)


adjs = torch.stack(list(map(adj, dataset))) # adj representing the connectivity 
sigs = torch.stack(list(map(sig, dataset))) # atom types as input
prop = torch.stack(list(map(target, dataset)))[:, 5] # HOMO property that needs to learn 


train_size, test_size = 5000, 1000
adjs_train,adjs_test  = adjs[:train_size], adjs[train_size:]
sigs_train,sigs_test  = sigs[:train_size], sigs[train_size:]
prop_train,prop_test  = prop[:train_size], prop[train_size:]



# print(len(dataset)) # 100
# print(adjs.size()) # 100,9,9 
# print(sigs.size()) # 100,9,5
# print(prop.size()) # 100

# print(dataset_smiles[0]) molecular name
# import sys
# sys.exit("some error message")


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


class GraphPooling:
    """
        Graph pooling layer
    """
    def __init__(self):
        self.pool= torch.sum

    def __call__(self, H):
        # -- multi-set pooling operator
        return self.pool(H,dim=-2)


class MyModel(nn.Module):
    """
        Regression  model
    """
    def __init__(self):
        super(MyModel, self).__init__()
        # -- initialize layers
        self.gc1 = GCN(5,3) # hidden layer dimension is 30
        self.fc1 = torch.nn.Linear(3,1)
        torch.nn.init.xavier_normal_(self.gc1.weight)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.pool = GraphPooling()
    def forward(self, A, h0):
        # print(A,h0)
        output = self.gc1(A,h0)
        # print(output)
        output = self.pool(output)
        # print(output)
        output = self.fc1(output)
        # print(output)
        return output


# import sys
# sys.exit("some error message")



"""
    Train
"""
# -- Initialize the model, loss function, and the optimizer
batch_size = 10
epochs = 201
model = MyModel()
MyLoss = torch.nn.MSELoss()
MyOptimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # 0.1~0.0001

# print out the model 
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data,param.data.requires_grad)
# model(adjs[0:3],sigs[0:3])
import sys
sys.exit("some error message")

batch_loss = []
train_loss = []
test_loss = []
# -- update parameters
# model.reset_parameters()
print('start training')

for epoch in tqdm(range(epochs)):
    for i in range(1):
        MyOptimizer.zero_grad()
        
        # -- predict
        pred = model(adjs_train[i*batch_size:(i+1)*batch_size], sigs_train[i*batch_size:(i+1)*batch_size])
        # print(pred.size())

        # -- loss
        # print(pred,prop[i*10:(i+1)*10] )
        loss = MyLoss(pred,prop_train[i*batch_size:(i+1)*batch_size])

        # -- optimize
        loss.backward()
        MyOptimizer.step()
        
        batch_loss.append(loss)
    if (epoch)%10==0:
        train_loss.append(MyLoss(model(adjs_train,sigs_train),prop_train))
        test_loss.append(MyLoss(model(adjs_test,sigs_test),prop_test))
train_loss = torch.stack(train_loss).detach().cpu().numpy()
test_loss = torch.stack(test_loss).detach().cpu().numpy()

# print(batch_loss)
# print(train_loss)
# -- plot loss
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 25,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
pylab.rcParams.update(params)
fig, ax = plt.subplots(figsize=(10,8))
lw=3
xx = np.linspace(0,200,21)
ax.plot(xx, train_loss, color = 'C0', label = 'trian')
ax.plot(xx, test_loss, color = 'C1', label = 'test')
# ax.scatter(x, y,c='C0',s=150, alpha=0.5, edgecolors=None,label = '')
# ax.set_xlim([,])
# ax.set_ylim([,])
ax.set_xlabel('epoch')
ax.set_ylabel('log scale error')
ax.set_yscale('log')
xticks = np.arange(11)*20
ax.set_xticks(xticks)
leg = ax.legend(loc='upper right',frameon = True)
leg.get_frame().set_edgecolor('black')
# fig.savefig('./{}.pdf'.format(),bbox_inches='tight')
fig.savefig('./results/qm9train1.png'.format(),bbox_inches='tight')



## 
prop_pred = model(adjs_test,sigs_test)
## print prediction results 

params = {'legend.fontsize': 25,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
pylab.rcParams.update(params)
fig, ax = plt.subplots(figsize=(10,8))
lw=3
xx = np.linspace(0,200,21)
ax.scatter(prop_pred.detach().cpu().numpy(), prop_test.detach().cpu().numpy(), color = 'C0')
# ax.scatter(x, y,c='C0',s=150, alpha=0.5, edgecolors=None,label = '')
ax.set_xlim([-0.4,0.0])
ax.set_ylim([-0.4,0.0])
ax.set_xlabel('prediction')
ax.set_ylabel('target')

# fig.savefig('./{}.pdf'.format(),bbox_inches='tight')
fig.savefig('./results/qm9train1_prediction.png'.format(),bbox_inches='tight')

