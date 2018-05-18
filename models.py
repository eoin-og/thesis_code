import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, num_labels, ns=32, seq_length=1024, latent_dim=100, ks=25, padding=11, stride=4, output_padding=1):
        super(Generator, self).__init__()
        
        mult = seq_length // 64
        min_net_size = ns

        ### Start block
        self.dc_noise = nn.ConvTranspose1d(latent_dim, mult * min_net_size, ks, stride=stride, padding=padding, output_padding=output_padding)
        self.bn_noise = nn.BatchNorm1d(min_net_size*mult)
        
        self.dc_labels = nn.ConvTranspose1d(sum(num_labels), mult * min_net_size, ks, stride=stride, padding=padding, output_padding=output_padding)
        self.bn_labels = nn.BatchNorm1d(min_net_size*mult)
        
        self.main = nn.Sequential()
        mult = mult * 2
        i = 1
        x = seq_length
        while x >= 64:
            self.main.add_module('Middle-CT [%d]' % i, nn.ConvTranspose1d(min_net_size*mult, min_net_size*(mult//2), ks, stride=stride, padding=padding, output_padding=output_padding))
            self.main.add_module('Middle-BN [%d]' % i, nn.BatchNorm1d(min_net_size*(mult//2)))
            self.main.add_module('Middle-RL [%d]' % i, nn.ReLU())
            mult = mult // 2
            i += 1
            x = x // 4

        self.main.add_module('End-CT', nn.ConvTranspose1d(min_net_size*mult, 1, ks, stride=stride, padding=padding, output_padding=output_padding))
        self.main.add_module('END-TH', nn.Tanh())

    def forward(self, noise, labels):
        x = F.relu(self.bn_noise(self.dc_noise(noise)))
        y = F.relu(self.bn_labels(self.dc_labels(labels)))
        x = torch.cat([x, y], 1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, num_labels, ns=32, seq_length=1024):
        super(Discriminator, self).__init__()
        
        kernel_size = 25
        min_net_size = ns
        self.cv_sample = nn.Conv1d(1, min_net_size // 2, kernel_size, 4, 1)
        self.bn_sample = nn.BatchNorm1d(min_net_size // 2)
        
        self.cv_labels = nn.Conv1d(sum(num_labels), min_net_size // 2, kernel_size, 4, 1)
        self.bn_labels = nn.BatchNorm1d(min_net_size // 2)
        
        self.main = nn.Sequential()
        mult = 1
        x = seq_length 
        i = 1
        while x >= 256:
            self.main.add_module('Middle-CT [%d]' % i, nn.Conv1d(min_net_size*mult, min_net_size*(mult*2), kernel_size, 4, 1))
            self.main.add_module('Middle-BN [%d]' % i, nn.BatchNorm1d(min_net_size * (mult*2)))
            self.main.add_module('Middle-RL [%d]' % i, nn.ReLU())
            mult = mult * 2
            x = x // 4
            i += 1

        self.main.add_module('End-CT', nn.Conv1d(min_net_size*mult, 1, 8, 4, 0))
        self.main.add_module('End-SG', nn.Sigmoid())

    def forward(self, sample, labels):
        x = F.relu(self.bn_sample(self.cv_sample(sample)))
        y = F.relu(self.bn_labels(self.cv_labels(labels)))
        x = torch.cat([x, y], 1)
        return self.main(x)
