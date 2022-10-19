from multiprocessing import Pool
import math, random, sys
import pickle
import argparse
from functools import partial
import torch
import numpy
from hgraph import MolGraph, common_atom_vocab, PairVocab
import rdkit

def unity(x):
    return x

def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

def tensorize(mol_batch, vocab,max_span_tree=True):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab,max_span_tree)
    return to_numpy(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--train_labels', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode', type=str, default='pair')
    parser.add_argument('--ncpu', type=int, default=8)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--max_span_tree', type=bool, default=True)
    args = parser.parse_args()
    
    with open(args.vocab) as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    args.vocab = PairVocab(vocab, cuda=False)

    pool = Pool(args.ncpu) 
    random.seed(1)
    
    if args.mode == 'single':
        #dataset contains single molecules
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[0] for line in f]
        with open(args.train_labels) as f:
            data_labels = [int(line.strip("\r\n ").split()[0]) for line in f]

        random.shuffle(data)
        random.shuffle(data_labels)

        batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        batches_labels = [data_labels[i : i + args.batch_size] for i in range(0, len(data_labels), args.batch_size)]
        
        #data
        func = partial(tensorize,vocab=args.vocab,max_span_tree=args.max_span_tree)
        all_data = pool.map(func, batches)
        
        print(len(all_data))
        num_splits = len(all_data) // 1000
        if num_splits < 1000 and num_splits >= 100:
            num_splits = len(all_data) // 100
        elif num_splits < 100:
            num_splits = len(all_data) // 10
        print(num_splits)
        le = (len(all_data) + num_splits - 1) // num_splits
        
        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('/scratch/gthurber_root/gthurber0/marcase/preprocess_mono/' +'tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
        
        
        #data_labels
        func = partial(unity)
        all_data = pool.map(func, batches_labels)
        
        print(len(all_data))
        num_splits = len(all_data) // 1000
        if num_splits < 1000 and num_splits >= 100:
            num_splits = len(all_data) // 100
        elif num_splits < 100:
            num_splits = len(all_data) // 10
        print(num_splits)
        le = (len(all_data) + num_splits - 1) // num_splits
        
        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('/scratch/gthurber_root/gthurber0/marcase/preprocess_mono/' + 'tensors_labels-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
                
    else:
        print(args.mode + ' is currently not a supported mode.')