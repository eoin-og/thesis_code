import os 
import time
import random
import pickle

import numpy as np
from scipy import signal
from scipy.io import wavfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable, grad

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils



def create_example(num_labels, length=128, noise=None):
    
    sample_range = length
    freqs = [261.626]
    wave_funcs = [np.sin, signal.sawtooth, signal.square]
    sps = 44100

    n = random.randint(0, 30)
    p = random.randint(10, 50)

    a = np.linspace(n, n+(length*4), num=length)
    loudness_factor = 1
    if len(num_labels) > 1:
    	loudness_factor = random.randint(1, num_labels[1])
    else:
    	loudness_factor = random.randint(1, 7)

    freq_index = random.randint(0, len(freqs) - 1)
    freq = freqs[freq_index]
    wave_index = random.randint(0, len(wave_funcs) - 1)
    wave_func = wave_funcs[wave_index]
    
    a = 2 * np.pi * a * freq / sps
    sample = wave_func(a)
    if noise:
    	sample = sample + np.random.uniform(low=-noise, high=noise, size=seq_length)

    return sample / loudness_factor, wave_index, loudness_factor - 1

def combine_things(labels, lot):
    comb = None
    for i, thing in enumerate(lot):
        comb_ = thing[labels[:, i]]
        try:
            comb = torch.cat([comb, comb_], 1)
        except TypeError:
            comb = comb_
    return comb

def create_rand_labels(num_labels, batch_size, replace=False):
    y = None
    for nl in num_labels:
        y_ = (torch.rand(batch_size, 1) * nl).type(torch.LongTensor)
        try:
            y = torch.cat([y, y_], 1)
        except:
            y = y_
    if replace:
        for j, i in enumerate(y):
            l, w = i.numpy()
            if w == 1 and l < 2:
                y[j][0] = random.randint(2, 6)
    return y

def test(D, G, real_label, fills, real_sample, batch_size, onehots, y_fake, y_real, num_labels, bce_loss, use_cuda):
    y_labels = Variable(combine_things(real_label, fills))
    y_labels = y_labels.cuda() if use_cuda else y_labels
    D_result = D(real_sample, y_labels).squeeze()
    D_real_loss = bce_loss(D_result, y_real)
    D_real_score = D_result.data.mean()
    
    z = Variable(torch.randn((batch_size, 100)).view(-1, 100, 1))
    y = create_rand_labels(num_labels, batch_size)
    y_label = Variable(combine_things(y, onehots))
    y_fill = Variable(combine_things(y, fills))

    if use_cuda:
    	z = z.cuda()
    	y_label = y_label.cuda()
    	y_fill = y_fill.cuda()

    G_result = G(z, y_label).detach()
    D_result = D(G_result, y_fill).squeeze()

    D_fake_loss = bce_loss(D_result, y_fake)
    D_fake_score = D_result.data.mean()
    
    if abs(D_real_score - D_fake_score) < 0.3:
        return False, False
    
    return D_real_score >= D_fake_score, D_real_score <= D_fake_score



def create_onehots(num_labels):
	onehots = []
	for num_label in num_labels:
	    onehot = torch.zeros(num_label, num_label)
	    onehots.append(onehot.scatter_(1, torch.LongTensor(range(num_label)).view(num_label,1), 1).view(num_label, num_label, 1))
	return onehots

def create_fills(num_labels, seq_length):
	fills = []
	for num_label in num_labels:
	    fill = torch.zeros([num_label, num_label, seq_length])
	    for i in range(num_label):
	        fill[i, i, :] = 1
	    fills.append(fill)
	return fills


def create_dataset(num_labels, seq_length, dataset_size, batch_size, noise):
	samples_list = []
	class_list = [[] for _ in num_labels]

	for _ in range(dataset_size):
	    example = create_example(num_labels, seq_length)
	    samples_list.append(example[0])
	    for i, cl_list in enumerate(class_list):
	    	cl_list.append(example[i+1])
	    
	data_tensor = torch.FloatTensor(samples_list).view(dataset_size, 1, -1)
	target_tensor = torch.FloatTensor(list(zip(*class_list)))

	dataset = torch.utils.data.TensorDataset(data_tensor, target_tensor)
	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_dataset(target_dirs, source_folder):
	samples_list = []
	class_list = []

	for i, td in enumerate(target_dirs):
		for file in os.listdir('{}/{}'.format(source_folder, td)):
			sample = wavfile.read(file)[1]
			samples_list.append(sample) ; class_list.append(i)

	    
	data_tensor = torch.FloatTensor(samples_list).view(dataset_size, 1, -1)
	target_tensor = torch.FloatTensor(list(zip(*class_list)))

	dataset = torch.utils.data.TensorDataset(data_tensor, target_tensor)
	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def gradient_penalty(gen_data, real_data, D, labels, fills, batch_size, lmda, cuda):
    gp_alpha = torch.rand(batch_size, 1, 1).expand(real_data.size())
    gp_alpha = gp_alpha.cuda() if cuda else gp_alpha
    interpolates = gp_alpha*real_data.data + (1 - gp_alpha)*gen_data.data 
    interpolates = interpolates.cuda() if cuda else interpolates
    interpolates = Variable(interpolates, requires_grad=True)
    y_labels = Variable(combine_things(labels, fills))
    y_labels = y_labels.cuda() if cuda else y_labels
    D_result = D(interpolates, y_labels)

    gradients = grad(outputs=D_result.sum(), inputs=interpolates, create_graph=True, retain_graph=True, only_inputs=True,
                        grad_outputs=torch.ones(D_result.size()).cuda() if cuda else torch.ones(D_result.size()))[0]
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lmda
    return gp

def set_gradients(net, value):
    for parameter in net.parameters():
        parameter.requires_grad = value


def norm(a):
    return(2*((a - a.min()) / (a.max() - a.min())) - 1)

def create_real_dataset(batch_size=32, dataset_size=2048, source_file=None, source_folder=None, targets=None, seq_length=16384, sect_length=1600):
    cl_list = []
    s_list = []

    if source_folder:
        for index, target in enumerate(targets):
            i = 0
            for file in os.listdir('{0}/{1}'.format(source_folder, target))[:sect_length]:
                sample = wavfile.read('{0}/{1}/{2}'.format(source_folder, target, file))[1]
                sample = norm(sample)
                if len(sample) < seq_length:
                    sample = np.pad(sample, (0,seq_length-len(sample)), 'constant', constant_values=(0))
                sample = sample[:seq_length]
                s_list.append(sample) ; cl_list.append(index)


    if source_file:
        sample = wavfile.read(source_file)[1]
        sample = 2*(sample - np.max(sample))/-np.ptp(sample)-1
        if len(sample) < seq_length:
            sample = np.pad(sample, (0,seq_length-len(sample)), 'constant', constant_values=(0))
        sample = sample[:seq_length]
        loudness = 1; wave = 1
        s_list.append(sample) ; loud_list.append(loudness) ; wave_list.append(wave)
    
    print(len(s_list), len(s_list[0]))
    data_tensor = torch.FloatTensor(s_list).view(dataset_size, 1, -1)
    target_tensor = torch.FloatTensor(cl_list)

    dataset = torch.utils.data.TensorDataset(data_tensor, target_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)