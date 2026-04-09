import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import norm
from torch import optim
from tqdm import tqdm
import torch
import seaborn as sns
import pyro
from pyro import distributions as distrs
from pyro.distributions import MultivariateNormal
import torchvision
from torch import nn
import math

def plot_2d_dots(dots, color='blue', label='None'):
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.scatter(dots[:, 0], dots[:, 1], s=1, c=color, label=label)


def get_parameters(mu=0., sigma=1.):
    train_mu = torch.Tensor([mu, mu]).requires_grad_(True)
    train_sigma = torch.Tensor([[sigma, 0.0],
                                [0.0, sigma]]).requires_grad_(True)
    return train_mu, train_sigma

def create_distr(mu, sigma):
    return distrs.MultivariateNormal(mu, sigma)


def sample(d, num):
    return d.sample(torch.Size([num]))

class MixtureDistribution:
    def __init__(self, p1, p2, w=0.5):
        self._p1 = p1
        self._p2 = p2
        self._w = w
        
    def sample(self, n):
        return torch.cat([sample(self._p1, int(n * self._w)), sample(self._p2, n - int(n * self._w))])
    
    def log_prob(self, x):
        return (self._w * self._p1.log_prob(x).exp() + (1. - self._w) * self._p2.log_prob(x).exp()).log()
    

def load_minst(used_MNIST_classes, device):
    def isin(ar1, ar2):
        return (ar1[..., None] == ar2).any(-1)


    train_set = torchvision.datasets.MNIST('./', train = True, download = True)
    idx = isin(train_set.train_labels, used_MNIST_classes)
    train_input  = train_set.train_data.view(-1, 1, 28, 28)[idx].to(device).float()
    train_target = train_set.train_labels[idx].to(device)

    test_set = torchvision.datasets.MNIST('./', train = False, download = True)
    idx = isin(test_set.train_labels, used_MNIST_classes)
    test_input = test_set.test_data.view(-1, 1, 28, 28)[idx].to(device).float()
    test_target = test_set.test_labels[idx].to(device)


    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std);
    test_input.sub_(mu).div_(std);
    return train_input, train_target, test_input, test_target

def entropy(target):
    probas = []
    for k in range(target.max() + 1):
        n = (target == k).sum().item()
        if n > 0: probas.append(n)
    probas = torch.tensor(probas).float()
    probas /= probas.sum()
    return - (probas * probas.log()).sum().item()

# Returns a triplet of tensors (a, b, c), where a and b contain each
# half of the samples, with a[i] and b[i] of same class for any i, and
# c is a 1d long tensor real classes
def create_image_pairs(train = False):
    ua, ub, uc = [], [], []

    if train:
        input, target = train_input, train_target
    else:
        input, target = test_input, test_target

    for i in used_MNIST_classes:
        used_indices = torch.arange(input.size(0), device = target.device)\
                            .masked_select(target == i.item())
        x = input[used_indices]
        x = x[torch.randperm(x.size(0))]
        hs = x.size(0) // 2
        ua.append(x.narrow(0, 0, hs))
        ub.append(x.narrow(0, hs, hs))
        uc.append(target[used_indices])

    a = torch.cat(ua, 0)
    b = torch.cat(ub, 0)
    c = torch.cat(uc, 0)
    perm = torch.randperm(a.size(0))
    a = a[perm].contiguous()

    if INDEPENDENT:
        perm = torch.randperm(a.size(0))
    b = b[perm].contiguous()

    return a, b, c


# Returns a triplet a, b, c where a are the standard MNIST images, c
# the classes, and b is a Nx2 tensor, with for every n:
#
#   b[n, 0] ~ Uniform(0, 10)
#   b[n, 1] ~ b[n, 0] + Uniform(0, 0.5) + c[n]
def create_image_values_pairs(train = False):
    ua, ub = [], []

    if train:
        input, target = train_input, train_target
    else:
        input, target = test_input, test_target

    m = torch.zeros(used_MNIST_classes.max() + 1, dtype = torch.uint8, device = target.device)
    m[used_MNIST_classes] = 1
    m = m[target]
    used_indices = torch.arange(input.size(0), device = target.device).masked_select(m)

    input = input[used_indices].contiguous()
    target = target[used_indices].contiguous()

    a = input
    c = target

    b = a.new(a.size(0), 2)
    b[:, 0].uniform_(0.0, 10.0)
    b[:, 1].uniform_(0.0, 0.5)

    if INDEPENDENT:
        b[:, 1] += b[:, 0] + \
                   used_MNIST_classes[torch.randint(len(used_MNIST_classes), target.size())]
    else:
        b[:, 1] += b[:, 0] + target.float()

    return a, b, c


def create_sequences_pairs(train = False):
    nb, length = 10000, 1024
    noise_level = 2e-2

    ha = torch.randint(nb_classes, (nb, ), device = device) + 1
    if INDEPENDENT:
        hb = torch.randint(nb_classes, (nb, ), device = device)
    else:
        hb = ha

    pos = torch.empty(nb, device = device).uniform_(0.0, 0.9)
    a = torch.linspace(0, 1, length, device = device).view(1, -1).expand(nb, -1)
    a = a - pos.view(nb, 1)
    a = (a >= 0).float() * torch.exp(-a * math.log(2) / 0.1)
    a = a * ha.float().view(-1, 1).expand_as(a) / (1 + nb_classes)
    noise = a.new(a.size()).normal_(0, noise_level)
    a = a + noise

    pos = torch.empty(nb, device = device).uniform_(0.0, 0.5)
    b1 = torch.linspace(0, 1, length, device = device).view(1, -1).expand(nb, -1)
    b1 = b1 - pos.view(nb, 1)
    b1 = (b1 >= 0).float() * torch.exp(-b1 * math.log(2) / 0.1) * 0.25
    pos = pos + hb.float() / (nb_classes + 1) * 0.5
    # pos += pos.new(hb.size()).uniform_(0.0, 0.01)
    b2 = torch.linspace(0, 1, length, device = device).view(1, -1).expand(nb, -1)
    b2 = b2 - pos.view(nb, 1)
    b2 = (b2 >= 0).float() * torch.exp(-b2 * math.log(2) / 0.1) * 0.25

    b = b1 + b2
    noise = b.new(b.size()).normal_(0, noise_level)
    b = b + noise

    return a, b, ha



######################################################################
from torch import nn
from torch.nn import functional as F

class NetForSequencePair(nn.Module):

    def feature_model(self):
        kernel_size = 10
        pooling_size = 3
        return  nn.Sequential(
            nn.Conv1d(1, self.nc, kernel_size = kernel_size),
            nn.AvgPool1d(pooling_size),
            nn.LeakyReLU(),
            nn.Conv1d(self.nc, self.nc, kernel_size = kernel_size),
            nn.AvgPool1d(pooling_size),
            nn.LeakyReLU(),
            nn.Conv1d(self.nc, self.nc, kernel_size = kernel_size),
            nn.AvgPool1d(pooling_size),
            nn.LeakyReLU(),
            nn.Conv1d(self.nc, self.nc, kernel_size = kernel_size),
            nn.AvgPool1d(pooling_size),
            nn.LeakyReLU(),
        )

    def __init__(self):
        super(NetForSequencePair, self).__init__()
        self.nc = 32
        self.nh = 256

        self.features_a = self.feature_model()
        self.features_b = self.feature_model()

        self.fully_connected = nn.Sequential(
            nn.Linear(2 * self.nc, self.nh),
            nn.ReLU(),
            nn.Linear(self.nh, 1)
        )

    def forward(self, a, b):
        a = a.view(a.size(0), 1, a.size(1))
        a = self.features_a(a)
        a = F.avg_pool1d(a, a.size(2))
        
        b = b.view(b.size(0), 1, b.size(1))
        b = self.features_b(b)
        b = F.avg_pool1d(b, b.size(2))

        x = torch.cat((a.view(a.size(0), -1), b.view(b.size(0), -1)), 1)
        return self.fully_connected(x)
    
class NetForImagePair(nn.Module):
    def __init__(self):
        super(NetForImagePair, self).__init__()
        self.features_a = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 5),
            nn.MaxPool2d(3), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size = 5),
            nn.MaxPool2d(2), nn.ReLU(),
        )

        self.features_b = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 5),
            nn.MaxPool2d(3), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size = 5),
            nn.MaxPool2d(2), nn.ReLU(),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(256, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, a, b):
        a = self.features_a(a).view(a.size(0), -1)
        b = self.features_b(b).view(b.size(0), -1)
        x = torch.cat((a, b), 1)
        return self.fully_connected(x)

######################################################################

class NetForImageValuesPair(nn.Module):
    def __init__(self):
        super(NetForImageValuesPair, self).__init__()
        self.features_a = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 5),
            nn.MaxPool2d(3), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size = 5),
            nn.MaxPool2d(2), nn.ReLU(),
        )

        self.features_b = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(256, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, a, b):
        a = self.features_a(a).view(a.size(0), -1)
        b = self.features_b(b).view(b.size(0), -1)
        x = torch.cat((a, b), 1)
        return self.fully_connected(x)
    
######################################################################

class NetForImageValuesPair(nn.Module):
    def __init__(self):
        super(NetForImageValuesPair, self).__init__()
        self.features_a = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 5),
            nn.MaxPool2d(3), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size = 5),
            nn.MaxPool2d(2), nn.ReLU(),
        )

        self.features_b = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(256, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, a, b):
        a = self.features_a(a).view(a.size(0), -1)
        b = self.features_b(b).view(b.size(0), -1)
        x = torch.cat((a, b), 1)
        return self.fully_connected(x)