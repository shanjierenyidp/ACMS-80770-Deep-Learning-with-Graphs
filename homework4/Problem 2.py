"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 4: Programming assignment
Problem 2
"""
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=UserWarning)
from torch import nn
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from chainer_chemistry import datasets
from chainer_chemistry.dataset.preprocessors.ggnn_preprocessor import GGNNPreprocessor
from tqdm import tqdm
# -- load data
class MolecularDataset(Dataset):
    def __init__(self, N, train=True):
        if train:
            start, end = 0, 100000
        else:
            start, end = 100000, 130000


        dataset, dataset_smiles = datasets.get_qm9(GGNNPreprocessor(kekulize=True),
                                                   return_smiles=True,
                                                   target_index=np.random.choice(range(133000)[start:end], N, False))

        self.atom_types = [6, 8, 7, 9, 1]
        self.V = 9

        self.adjs = torch.stack(list(map(self.adj, dataset))) # adj representing the connectivity 
        self.sigs = torch.stack(list(map(self.sig, dataset))) # atom types as input
        self.prop = torch.stack(list(map(self.target, dataset)))[:, 5] # HOMO property that needs to learn 
        self.prop_2 = torch.stack(list(map(self.target_2, dataset_smiles)))  

    def target_2(self, smiles):
        """
            compute the number of hydrogen-bond acceptor atoms
        :param smiles: smiles molecular representation
        :return:
        """
        mol = Chem.MolFromSmiles(smiles)

        return torch.tensor(Chem.rdMolDescriptors.CalcNumHBA(mol))

    def adj(self, x):
        x = x[1]
        adjacency = np.zeros((self.V, self.V)).astype(float)
        adjacency[:len(x[0]), :len(x[0])] = x[0] + 2 * x[1] + 3 * x[2]
        return torch.tensor(adjacency)

    def sig(self, x):
        x = x[0]
        atoms = np.ones((self.V)).astype(float)
        atoms[:len(x)] = x
        out = np.array([int(atom == atom_type) for atom_type in self.atom_types for atom in atoms]).astype(float)
        return torch.tensor(out).reshape(5, len(atoms)).T

    def target(self, x):
        """
            return Highest Occupied Molecular Orbital (HOMO) energy
        :param x:
        :return:
        """
        x = x[2]
        return torch.tensor(x)

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, item):
        return self.adjs[item], self.sigs[item], self.prop[item], self.prop_2[item]

class kernel:
    def __init__(self, K, R, d, J, lamb_max):
        # -- filter properties
        self.R = float(R)
        self.J = J
        self.K = K 
        self.d = d
        self.lamb_max = torch.tensor(lamb_max)

        # -- Half-Cosine kernel
        # self.a = ...
        # self.g_hat = lambda lamb: ...
        
        # dialiation factor 
        self.a = self.R*np.log10(self.lamb_max.detach().numpy())/(self.J-self.R+1)
        self.g_hat = lambda lamb:sum([ele*np.cos(2*np.pi*k*(lamb/self.a+0.5))*1*(-lamb>=0 and -lamb<self.a) for k,ele in enumerate(self.d)])

    def wavelet(self, lamb, j):
        """
            constructs wavelets ($j\in [2, J]$).
        :param lamb: eigenvalue (analogue of frequency).
        :param j: filter index in the filter bank.
        :return: filter response to input eigenvalues.
        """
        # compute filter at this sacle:
        assert(j>=2 and j<=self.J), 'j must be between [2,J]'
        
        # calculate critical value 
        lamb_star = 10**(self.a*((j-1)/self.R-1))
        if  lamb < lamb_star:
            # print(lamb)
            lamb = lamb_star
        return torch.tensor(self.g_hat(np.log10(lamb) - self.a*(j-1)/self.R))
    
    def scaling(self, lamb):
        """
            constructs scaling function (j=1).
        :param lamb: eigenvalue (analogue of frequency).
        :return: filter response to input eigenvalues.
        """
        return torch.tensor(np.sqrt(self.R*self.d[0]**2+ self.R/2*sum([ele**2 for ele in self.d])-sum([abs(self.wavelet(lamb,i))**2 for i in range(2,self.J+1)])))

class scattering(nn.Module):
    def __init__(self, J, L, V, d_f, K, d, R, lamb_max):
        super(scattering, self).__init__()

        # -- graph parameters
        self.n_node = V
        self.n_atom_features = d_f

        # -- filter parameters
        self.K = K
        self.d = d
        self.J = J
        self.R = R
        self.lamb_max = lamb_max
        self.filters = kernel(K=1, R=3, d=[0.5, 0.5], J=8, lamb_max=2)

        # -- scattering parameters
        self.L = L  # number of layers

    def compute_spectrum(self, W):
        """
            Computes eigenvalues of normalized graph Laplacian.
        :param W: tensor of graph adjacency matrices.
        :return: eigenvalues of normalized graph Laplacian
        """

        # -- computing Laplacian
        # L = ...
        # W = W+torch.eye(W.size()[1])
        L = torch.diag(W.sum(1)) - W

        # -- normalize Laplacian
        diag = W.sum(1)
        dhalf = torch.diag_embed(1. / torch.sqrt(torch.max(torch.ones(diag.size()), diag)))
        # L = ...
        L = torch.chain_matmul(dhalf,L,dhalf)

        # -- eig decomposition
        E, V = torch.symeig(L, eigenvectors=True)
        #
        # print(torch.min(E))
        return abs(E), V           

    def filtering_matrices(self, W):
        """
            Compute filtering matrices (frames) for spectral filters
        :return: a collection of filtering matrices of each wavelet kernel and the scaling function in the filter-bank.
        """

        filter_matrices = []
        E, V = self.compute_spectrum(W)
        # print('eigen values', E)
        # -- scaling frame
        # filter_matrices.append(V @ ... @ V.T)

        filter_matrices.append(V @ torch.diag(torch.tensor([self.filters.scaling(ele) for ele in E])) @ V.T)
        # filter_matrices.append(torch.chain_matmul(V,torch.diag([self.filters.scaling(ele) for ele in E]),V.T))


        # -- wavelet frame
        for j in range(2, self.J+1):
            # filter_matrices.append(V @ ... @ V.T)
            filter_matrices.append(V @ torch.diag(torch.tensor([self.filters.wavelet(ele,j) for ele in E])) @ V.T)
            # filter_matrices.append(torch.chain_matmul(V,torch.diag([self.filters.wavelet(ele,j) for ele in E]),V.T))

        return torch.stack(filter_matrices)

    def forward(self, W, f):
        """
            Perform wavelet scattering transform
        :param W: tensor of graph adjacency matrices.
        :param f: tensor of graph signal vectors. f has shape of  num of vertices * num of channels
        :return: wavelet scattering coefficients
        """

        # -- filtering matrices
        g = self.filtering_matrices(W)

        # --
        U_ = [f]

        # -- zero-th layer
        # S = ...  # S_(0,1)
        S = torch.mean(f,dim = 0)

        for l in range(self.L):
            U = U_.copy()
            U_ = []

            for f_ in U:
                for g_j in g:

                    U_.append(abs(g_j@f_))

                    # -- append scattering feature S_(l,i)
                    S_li = torch.mean(abs(g_j@f_),dim=0)
                    S = torch.cat((S, S_li))

        return S

# -- initialize scattering function
scat = scattering(L=2, V=9, d_f=5, K=1, R=3, d=[0.5, 0.5], J=8, lamb_max=2)

# -- load data
data_size = 6000
# data_size = 6000
data = MolecularDataset(N=data_size)
# print(data[0][1].size())
# -- Compute scattering feature maps
feature_maps = []
from tqdm import tqdm
for i in tqdm(range(data_size)):
    feature_maps.append(scat(data[i][0],data[i][1]))
feature_maps = torch.stack(feature_maps)

# -- PCA projection
from sklearn.decomposition import PCA
X = feature_maps.detach().numpy()
pca = PCA(n_components=2)
latent = pca.fit_transform(X)

# -- plot feature space
target2_list = torch.stack([data[i][3] for i in range(data_size)])
target2_list_np = np.array(target2_list.detach().numpy())

# save
np.savez('data/pca',latent,target2_list_np)
# load
latent = np.load('data/pca.npz')['arr_0']
target2_list_np = np.load('data/pca.npz')['arr_1']

#figure single
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 25,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
pylab.rcParams.update(params)
fig, ax = plt.subplots(figsize=(10,8))
lw=3

map = ax.scatter(latent[:,0], latent[:,1],c=target2_list_np,s=150, alpha=1, edgecolors=None,label = None)

# ax.set_xlim([0,lamb_max])
# ax.set_ylim([,])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
fig.colorbar(map, ax =ax)
# leg = ax.legend(loc='upper right',frameon = True)
# leg.get_frame().set_edgecolor('black')
fig.savefig('results/p{}_pcaplot.pdf'.format(2),bbox_inches='tight')
fig.savefig('results/p{}_pcaplot.png'.format(2),bbox_inches='tight')

## training a 2-layer neural network to predict HOMO from scattering representations



# My_inputs = feature_maps
# My_outputs = torch.stack([data[i][2] for i in range(data_size)])
# print('input size', My_inputs.size() , 'output size', My_outputs.size())
# #save data
# torch.save({'inputs':My_inputs,
#             'outputs':My_outputs},
#            './data/Training_data.pt')


#load data
data_pack = torch.load('./data/Training_data.pt')
My_inputs = torch.tensor(data_pack['inputs']).float()
My_outputs = torch.tensor(data_pack['outputs']).float()
print('input size', My_inputs.size() , 'output size', My_outputs.size())



# # -- PCA projection
# latent_dim = 20
# from sklearn.decomposition import PCA
# X = My_inputs.detach().numpy()
# pca = PCA(n_components=latent_dim)
# latent = pca.fit_transform(X)
# print(np.cumsum(pca.explained_variance_ratio_))
# My_inputs = torch.tensor(latent).float()

# normalizing parameters
class My_Normalizer():
    """Trajectory dataset"""
    def __init__(self, data, mode = 'mean_std'):
        self.mode = mode
        if mode == 'mean_std':
            self.norm_params = [torch.mean(data, dim=0), torch.std(data, dim=0)]
        elif mode =='max_min':
            self.norm_params = [torch.max(data, dim=0)[0], torch.min(data, dim=0)[0]]
        elif mode == 'none':
            self.norm_params = None
        # print(self.norm_params)
    def normalize(self,x):
            return self.element_normalize(x, self.norm_params, mode = self.mode)
    def denormalize(self,x):
            return self.element_denormalize(x,self.norm_params, mode = self.mode)
 
    @staticmethod
    def element_normalize(new_data, norm_params, mode = 'mean_std'):
        if mode == 'max_min':
            data_max, data_min = norm_params
            normalized = (new_data - torch.tensor(data_min)) / (torch.tensor(data_max) - torch.tensor(data_min)) * 1 - 0
            normalized[torch.isnan(normalized)] = 0.0
        elif mode == 'mean_std':
            data_mean, data_std = norm_params
            normalized = (new_data - torch.tensor(data_mean)) / torch.tensor(data_std) 
            normalized[torch.isnan(normalized)] = 0.0
        elif mode == 'none':
            normalized = new_data
        return normalized
    @staticmethod
    def element_denormalize(new_data, norm_params, mode = 'mean_std'):
        if mode ==  'max_min':
            data_max, data_min = norm_params
            denormalized = (new_data+0)*(torch.tensor(data_max) - torch.tensor(data_min))/1 + torch.tensor(data_min)
        elif mode == 'mean_std':
            data_mean, data_std = norm_params
            denormalized = new_data*torch.tensor(data_std) + torch.tensor(data_mean)
        elif mode == 'none':
            denormalized = new_data
        return denormalized

input_normalizer = My_Normalizer(My_inputs, mode = 'max_min')
# input_normalizer = My_Normalizer(My_inputs, mode = 'max_min')
output_normalizer = My_Normalizer(My_outputs, mode = 'none')

class My_Dataset(Dataset):
    """Trajectory dataset"""
    def __init__(self, my_input, my_output, normalizers = [input_normalizer,output_normalizer]):
        input_normalizer, output_normalizer = normalizers[0], normalizers[1]
        if not len(my_input) == len(my_output):
            print("Error, input output doesnt match")
        self.my_input = input_normalizer.normalize(my_input)
        self.my_output = output_normalizer.normalize(my_output)
    def __len__(self):
        return len(self.my_input)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        datainput = self.my_input[idx]
        datalabel = self.my_output[idx]
        sample = {'input': datainput ,'output': datalabel}
        return sample
    
batch_size = 500
device = 'cuda'
train_size = 5000 # data_size =5000
test_size = 1000 # 1000
train_data = {'inputs': My_inputs[:train_size],'outputs': My_outputs[:train_size]}
test_data = {'inputs': My_inputs[-test_size:],'outputs': My_outputs[-test_size:]}

print(My_inputs.size(),My_outputs.size()) # input size torch.Size([6000, 365]) output size torch.Size([6000])
train_dataset = My_Dataset(train_data['inputs'],train_data['outputs'].unsqueeze(1))
test_dataset = My_Dataset(test_data['inputs'],test_data['outputs'].unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1,shuffle=True, num_workers=0)


# import sys
# sys.exit('debug')

## check normalization 
# rec_prediction = input_normalizer.denormalize(input_normalizer.normalize(train_data['inputs']))
# print(torch.max(rec_prediction - train_data['inputs']))
# rec_prediction = output_normalizer.denormalize(output_normalizer.normalize(train_data['outputs']))
# print(torch.max(rec_prediction - train_data['outputs']))
# print(train_data['outputs'])
# print(output_normalizer.normalize(train_data['outputs']))


import torch.nn.functional as F
class My_model(torch.nn.Module):
    def __init__(self,in_channels, out_channels,hidden_channels):
        super(My_model,self).__init__()
        channels = np.concatenate([[in_channels],hidden_channels,[out_channels]])
        self.channels = [int(ele) for ele in channels]
        self.layers= torch.nn.Sequential()
        for i in range(len(self.channels)-1):
            self.layers.add_module('conv_{:d}'.format(i),nn.Linear(self.channels[i], self.channels[i+1]))
            if i<len(self.channels)-1-1:
                self.layers.add_module('relu_{:d}'.format(i), torch.nn.ReLU()) # ReLU, Sigmoid, LeakyReLU
    def forward(self,x):
        x = self.layers(x)
        return x
    
def train(model, device, train_loader, optimizer, criterion, scheduler = None):
    # define trainloader
    model.train()
    train_ins_error = []
    for batch_idx, sample in enumerate(train_loader):
        # print(batch_idx)
        inputs,labels = sample['input'], sample['output']
        inputs,labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # print('pan debug output size:', outputs.size())
        # print('pan debug label size:', labels.size())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
        train_ins_error.append(loss)
    return torch.mean(torch.stack(train_ins_error))

def test(model, my_input, my_output):
    model.to('cpu')
    my_predict = output_normalizer.denormalize(model(input_normalizer.normalize(my_input)))
    test_error  = torch.norm((my_predict-my_output))# / torch.norm(my_output)
    test_error_mean  = torch.mean(test_error)
    model.to(device)
    return my_predict, test_error, test_error_mean
    
# model = My_model(365,1,[365,256,128,64,32,16,8,4]) # input 365, output 1
model = My_model(My_inputs.size()[-1],1,[365]) # input 365, output 1
model.to(device)  # push the nework to the assigned device 
# for name, param in model.state_dict().items():
#     print(name, param.size())
print(model)

lr = 0.0001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.001)
epoches = 2000 #2000

##############################################################################################
train_ins_error = []
train_error = []
test_error = []
test_stepsize = 10
plot_stepsize = 500
for epoch in tqdm(range(epoches)):
    temp_error= train(model, device, train_loader, optimizer,criterion)
    train_ins_error.append(temp_error)

    if (epoch+1)%test_stepsize ==0:
        # print(scheduler.get_last_lr())
        _,train_error_temp, train_error_mean_temp = test(model, train_data['inputs'], train_data['outputs'])
        _,test_error_temp , test_error_mean_temp = test(model, test_data['inputs'], test_data['outputs'])
        train_error.append(train_error_mean_temp.detach().cpu().numpy())
        test_error.append(test_error_mean_temp.detach().cpu().numpy())
    if (epoch+1)%plot_stepsize ==0:
        # print(scheduler.get_last_lr())
        # print('plotting')
        fig,ax = plt.subplots(figsize = (10,8))
        # ax.plot(torch.stack(train_ins_error).detach().cpu().numpy(),'r')
        # print(train_error)
        # print(test_error)
        ax.plot(np.arange(len(train_error))*test_stepsize,train_error,'r')
        ax.plot(np.arange(len(test_error))*test_stepsize,test_error,'g')
        ax.set_yscale("log")
        fig.savefig('curve.png')

train_ins_error = torch.stack(train_ins_error)

# saving trained results
torch.save({
            'epoch': epoches,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_ins_error,
            }, './data/Training_result.pt')
##########################################################################################


## load in trained data 
checkpoint_pack = torch.load('./data/Training_result.pt')
train_ins_error = checkpoint_pack['loss']
# figure single
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 25,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
pylab.rcParams.update(params)
fig, ax = plt.subplots(figsize=(10,8))
lw=3

ax.plot(train_ins_error.detach().cpu().numpy(),color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='')
# ax.set_xlim([,])
# ax.set_ylim([,])
ax.set_xlabel('epoch')
ax.set_ylabel('log scale error')
ax.set_yscale('log')
# leg = ax.legend(loc='upper right',frameon = True)
# leg.get_frame().set_edgecolor('black')
fig.savefig('results/p{}_train_error.pdf'.format(2),bbox_inches='tight')
fig.savefig('results/p{}_train_error.png'.format(2),bbox_inches='tight')





model.load_state_dict(checkpoint_pack['model_state_dict'])
model.to('cpu')
# prediction = model(test_data['inputs'])
fit_prediction = output_normalizer.denormalize(model(input_normalizer.normalize(train_data['inputs'])))
prediction = output_normalizer.denormalize(model(input_normalizer.normalize(test_data['inputs'])))

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 25,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
pylab.rcParams.update(params)
fig, ax = plt.subplots(figsize=(10,8))
lw=3
ax.plot(train_data['outputs'].detach().cpu().numpy(),train_data['outputs'].detach().cpu().numpy(),color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='ground truth')
ax.scatter(train_data['outputs'].detach().cpu().numpy(),fit_prediction.detach().cpu().numpy(),color = 'C1',marker = 'o',s=10,linestyle='solid',linewidth=lw,alpha=1,label='test')
# ax.set_xlim([,])
# ax.set_ylim([,])
ax.set_xlabel('epoch')
ax.set_ylabel('error')
# ax.set_yscale('log')
leg = ax.legend(loc='upper left',frameon = True)
leg.get_frame().set_edgecolor('black')
fig.savefig('results/p{}_fit_error.pdf'.format(2),bbox_inches='tight')
fig.savefig('results/p{}_fit_error.png'.format(2),bbox_inches='tight')

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 25,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
pylab.rcParams.update(params)
fig, ax = plt.subplots(figsize=(10,8))
lw=3
xx, yy = prediction.detach().cpu().numpy(),test_data['outputs'].detach().cpu().numpy()
ax.plot(xx,xx,color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='ground truth')
# ax.scatter(test_data['outputs'].detach().cpu().numpy(),prediction.detach().cpu().numpy(),color = 'C1',marker = 'o',s=10,linestyle='solid',linewidth=lw,alpha=1,label='test')
ax.scatter(xx,yy,color = 'C1',marker = 'o',s=10,linestyle='solid',linewidth=lw,alpha=1,label='test')
# ax.set_xlim([,])
# ax.set_ylim([,])
ax.set_xlabel('prediction')
ax.set_ylabel('target')
# ax.set_yscale('log')
leg = ax.legend(loc='upper left',frameon = True)
leg.get_frame().set_edgecolor('black')
fig.savefig('results/p{}_test_error.pdf'.format(2),bbox_inches='tight')
fig.savefig('results/p{}_test_error.png'.format(2),bbox_inches='tight')
