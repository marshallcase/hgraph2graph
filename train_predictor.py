#train predictor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import rdkit
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm

from hgraph import *
from hgraph.predict import HierPredict

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_model',type=bool,default=False)
parser.add_argument('--model')
parser.add_argument('--seed', type=int, default=7)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--step_beta', type=float, default=0.001)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=25000)
parser.add_argument('--print_iter', type=int, default=50)

parser.add_argument('--save_iter', type=int, default=5000)
parser.add_argument('--train', required=True)
parser.add_argument('--train_labels',required=True)
parser.add_argument('--run_test',type=bool,default=True)
parser.add_argument('--test')
parser.add_argument('--test_labels')
parser.add_argument('--label_size',type=int,default = 2)
parser.add_argument('--separate_predict',default=False)

args = parser.parse_args()
print(args)

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)

#get model
if args.separate_predict == True:
    encoder = HierVAE(args).cuda()
    model = HierPredict(args).cuda()
else:
    model = HierVAE(args).cuda()

#fix random state
torch.manual_seed(args.seed)
random.seed(args.seed)

#initialize weights
if args.separate_predict == True:
    for param in encoder.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)   
else:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)        

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

#load model if args.load_model == True
if args.load_model:
    print('continuing from checkpoint ' + args.model)
    model_state, optimizer_state, total_step, beta = torch.load(args.model)
    
    print(args.separate_predict)
    if args.separate_predict == True:
        encoder.load_state_dict(model_state)
    else:
        #initialize weights in model.predict that don't exist from pre-training
        for key in model.predict.state_dict().keys():
            model_state['predict.' + key] = model.predict.state_dict()[key]

        model.load_state_dict(model_state)


else:
    total_step = beta = 0

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

meters = np.array([])
meters_list = list(meters)
validation_list = list()
total_step = 0

#main train loop
for epoch in range(args.epoch):
    random.seed(args.seed)
    dataset_x = DataFolder(args.train, args.batch_size,shuffle = False)
    dataset_y = DataFolder(args.train_labels, args.batch_size,shuffle = False)
    dataset_x.data_files = ['tensors-'+str(i)+'.pkl' for i in range(len(dataset_x.data_files))]
    dataset_y.data_files = ['tensors_labels-'+str(i)+'.pkl' for i in range(len(dataset_y.data_files))]
    model.train()
    for batch_x,batch_y in zip(dataset_x,dataset_y):
        total_step += 1
        model.zero_grad()
        if args.separate_predict == True:
            latent = encoder(*batch_x, beta=beta,decode=False,predict=True) 
            y_pred = model(latent)
        else:
            y_pred = model(*batch_x,beta=beta,decode=False,predict=True)
            
        y_true = torch.Tensor([int(y) for y in batch_y]).cuda()
        y_true = y_true.type(torch.LongTensor).cuda()
        loss = criterion(y_pred,y_true)
        accuracy = torch.sum(torch.argmax(y_pred, dim=1).cuda() == y_true)/len(y_true)
        # loss = Variable(loss, requires_grad = True)
        loss.backward()
        optimizer.step()

        meters = np.array([loss.item(),accuracy.cpu()])
        meters_list.append(meters)

        if total_step % args.print_iter == 0:
            print("[%d] Beta: %.3f, loss: %.3f, accuracy: %.3f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], param_norm(model), grad_norm(model)))
            sys.stdout.flush()
            meters *= 0
        
        if total_step % args.save_iter == 0:
            ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
            torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{total_step}"))

        # if total_step % args.anneal_iter == 0:
        #     scheduler.step()
        #     print("learning rate: %.6f" % scheduler.get_lr()[0])

        if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
            beta = min(args.max_beta, beta + args.step_beta)
    
    #"validation" set
    if args.run_test:
        model.eval()
        dataset_x = DataFolder(args.test, args.batch_size,shuffle = False)
        dataset_y = DataFolder(args.test_labels, args.batch_size,shuffle = False)
        dataset_x.data_files = ['tensors-'+str(i)+'.pkl' for i in range(len(dataset_x.data_files))]
        dataset_y.data_files = ['tensors_labels-'+str(i)+'.pkl' for i in range(len(dataset_y.data_files))]
        random.seed()
        i=0
        accuracy_list = list()
        for batch_x,batch_y in zip(dataset_x,dataset_y):
            batch_x0 = batch_x
            batch_y0 = batch_y
            if args.separate_predict == True:
                latent = encoder(*batch_x, beta=beta,decode=False,predict=True) 
                y_pred = model(latent)
            else:
                y_pred = model(*batch_x,beta=beta,decode=False,predict=True)
                
            y_pred = torch.argmax(y_pred, dim=1)
            y_true = torch.Tensor([int(y) for y in batch_y0]).cuda()
            y_true = y_true.type(torch.LongTensor).cuda()
            accuracy_list.append((torch.sum(y_pred == y_true)/len(y_pred)).item())
            i += 1


        print('Accuracy on validation set: %.3f' % np.average(accuracy_list))
        validation_list.append((torch.sum(y_pred == y_true)/len(y_pred)).item())
