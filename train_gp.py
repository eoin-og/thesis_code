import random
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn 
import torch.optim as optim
from torch.autograd import Variable

from utils import *
from models import *


# hyperparameters
batch_size = 32
num_epoch = 100
dlr = 0.0002
glr = 0.005
beta1 = 0.5
min_net_size = 10
phi = 4

# variables
use_cuda = True
allow_freezing = True
real_dataset = True
t
arget_dirs = 'one two five seven'.split()
sect_length = 512 
num_labels = [len(target_dirs)] #[3, 7]
seq_length = 16384
dataset_size = sect_length*len(target_dirs)
latent_dim = 100
noise = 0.05
lmda = 0.1

# create dataset
if real_dataset:
    num_labels = [len(target_dirs)]
    dataloader = create_real_dataset(batch_size=batch_size, 
                                     dataset_size=dataset_size, 
                                     source_folder='speech_data',
                                     targets=target_dirs,
                                     sect_length=sect_length,
                                     seq_length=seq_length)
else:
    num_labels = [3, 7]
    dataloader = create_dataset(num_labels, seq_length=seq_length, dataset_size=dataset_size, batch_size=batch_size, noise=noise)

D = Discriminator(num_labels, ns=min_net_size*phi, seq_length=seq_length)
G = Generator(num_labels, ns=min_net_size, seq_length=seq_length)
loss_function = nn.BCELoss()
D_optimizer = optim.Adam(D.parameters(), lr=dlr, betas=(beta1, 0.999))
G_optimizer = optim.Adam(G.parameters(), lr=glr, betas=(beta1, 0.999))


onehots = create_onehots(num_labels)
fills = create_fills(num_labels, seq_length)
y_real = Variable(torch.ones(batch_size))
y_fake = Variable(torch.zeros(batch_size)) 
start_time = time.time()
freeze_generator, freeze_discriminator = False, False
fakes = []
d_count, g_count = 0, 0

if use_cuda:
    D = D.cuda()
    G = G.cuda()
    y_real = y_real.cuda()
    y_fake = y_fake.cuda()

for epoch in range(num_epoch):
    for i, (real_sample, real_label) in enumerate(dataloader, 0):
        
        real_label = real_label.type(torch.LongTensor).view(batch_size, len(num_labels))
        real_sample = Variable(real_sample).cuda() if use_cuda else Variable(real_sample)
        
        
        if not freeze_discriminator:
            
            D.zero_grad() ; G.zero_grad() ; set_gradients(D, True) ; set_gradients(G, False)
            d_count += 1
            y_labels = Variable(combine_things(real_label, fills))
            y_labels = y_labels.cuda() if use_cuda else y_labels

            D_result = D(real_sample, y_labels).squeeze()
            D_real_loss = D_result.mean()
            D_real_score = D_result.data.mean() 
            
            ################################

            z = Variable(torch.randn((batch_size, latent_dim, 1)))
            y = real_label
            y_label = Variable(combine_things(y, onehots))
            y_fill = Variable(combine_things(y, fills))
            
            if use_cuda:
                z = z.cuda()
                y_label = y_label.cuda()
                y_fill = y_fill.cuda()

            G_result = G(z, y_label).detach()
            D_result = D(G_result, y_fill).squeeze()
            D_fake_loss = D_result.mean()
            D_fake_score = D_result.data.mean()


            gp = gradient_penalty(G_result, real_sample, D, real_label, fills, batch_size, lmda, use_cuda)

            D_train_loss = D_fake_loss - D_real_loss + gp

            D_train_loss.backward()
            D_optimizer.step()
            
        if not freeze_generator:
            G.zero_grad() ; D.zero_grad() ; set_gradients(D, False) ; set_gradients(G, True)
            g_count += 1

            z = Variable(torch.randn((batch_size, latent_dim, 1)))
            y = real_label #create_rand_labels(num_labels, batch_size)
            y_label = Variable(combine_things(y, onehots))
            y_fill = Variable(combine_things(y, fills))
            
            if use_cuda:
                z = z.cuda()
                y_label = y_label.cuda()
                y_fill = y_fill.cuda()

            G_result = G(z, y_label)
            D_result = D(G_result, y_fill).squeeze()

            G_train_loss = -D_result.mean()
            G_score = D_result.data.mean()

            G_train_loss.backward()
            G_optimizer.step() 
        
        if allow_freezing:
            freeze_discriminator, freeze_generator = test(D, G, real_label, fills, real_sample, batch_size, 
                                                            onehots, y_fake, y_real, num_labels, loss_function, use_cuda)
    
    
    print('[{0}/{1}] -- D real: {2} -- D fake: {3} -- G: {4} --- {5}:{6} -- time: {7}'.format(epoch, num_epoch,
                                                                                          D_real_score, D_fake_score, G_score,
                                                                                          d_count, g_count, time.time() - start_time))
    fakes.append({'results':G_result.cpu(), 'time':time.time() - start_time, 'epoch':epoch, 
                    'drs':D_real_score, 'dfs':D_fake_score, 'gs':G_score, 'dc':d_count, 'gc':g_count, 'labels':real_label})
    d_count, g_count = 0, 0
    pickle.dump(fakes, open('results.p', 'wb'))