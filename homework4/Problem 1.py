"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame

Homework 4: Programming assignment
Problem 1
"""
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=UserWarning)


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
        assert(j>=2 and J<=self.J), 'j must be between [2,J]'
        return self.g_hat(np.log10(lamb) - self.a*(j-1)/self.R)
    
    def scaling(self, lamb):
        """
            constructs scaling function (j=1).
        :param lamb: eigenvalue (analogue of frequency).
        :return: filter response to input eigenvalues.
        """
        return np.sqrt(self.R*self.d[0]**2+ self.R/2*sum([ele**2 for ele in self.d])-sum([abs(self.wavelet(lamb,i))**2 for i in range(2,self.J+1)]))


# -- define filter-bank
lamb_max = 2
J = 8
filter_bank = kernel(K=1, R=3, d=[0.5, 0.5], J=J, lamb_max=lamb_max)
# xx = np.linspace(0,lamb_max,1001)[1:]
# print(np.array([filter_bank.wavelet(ele,2) for ele in xx]))
# print(np.array([filter_bank.scaling(ele) for ele in xx]))

# import sys
# sys.exit('pan debug stop')
# -- plot filters

#figure single
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 25,'axes.labelsize': 25,'axes.titlesize':25,'xtick.labelsize':25,'ytick.labelsize':25}
pylab.rcParams.update(params)
fig, ax = plt.subplots(figsize=(10,8))
lw=3

xx = np.linspace(0,lamb_max,1001)[1:]
for i in range(1,J+1):
    if i ==1:
        yy = np.array([filter_bank.scaling(ele) for ele in xx])
        ax.plot(xx, yy,color = 'C{:d}'.format(i-1),linestyle='solid',linewidth=lw,alpha=1,label='filter{:d}'.format(i))
    else:
        yy = np.array([filter_bank.wavelet(ele,i) for ele in xx])
        ax.plot(xx, yy,color = 'C{:d}'.format(i-1),linestyle='solid',linewidth=lw,alpha=1,label='filter{:d}'.format(i))
ax.set_xlim([0,lamb_max])


# ax.scatter(x, y,c='C0',s=150, alpha=0.5, edgecolors=None,label = '')
# ax.set_ylim([,])
ax.set_xlabel('lambda')
ax.set_ylabel('response')
leg = ax.legend(loc='center left',frameon = True)
leg.get_frame().set_edgecolor('black')
fig.savefig('results/p{}_filterbankplot.pdf'.format(1),bbox_inches='tight')
fig.savefig('results/p{}_filterbankplot.png'.format(1),bbox_inches='tight')

