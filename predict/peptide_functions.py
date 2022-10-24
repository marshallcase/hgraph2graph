#generate in-library peptide sequences
import pandas as pd
import numpy as np
import os
from pandas import DataFrame
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker
import random
from sequence_to_smiles import *
import itertools

aa_df = pd.read_excel('AA_smiles_dict.xlsx')
aa_df = aa_df.set_index('1 letter')
aa_set = set(aa_df.drop(['Z','C','P']).index.values)

def flatten(l):
    return [item for sublist in l for item in sublist]

def getPeptide(m=None,n=None,o=None,n_seq=1,AAs=aa_set) -> str:
    '''
    getPeptide: return randomized cyclic peptide like Kong2020
    Inputs:
        m: int
        n: int
        o: int (3<= m+n+o <= 8)
        n_seq: int, # of sequences to generate
        AAs: set of amino acids to be used
    Outputs:
        randomized peptide sequence(s)
    '''
    if (m is None) and (n is None) and (o is None):
        length_random=True
        
    peptides = []
    for i in range(n_seq):
        if length_random:
            total_length = random.randint(3,8)
            m = random.randint(1,total_length-2)
            n = random.randint(1,total_length-1-m)
            o = random.randint(1,total_length-m-n)
            x = [m,n,o]
            random.shuffle(x)
            m,n,o = [i for i in x]
        peptides.append("".join([str(i) for i in (random.sample(AAs,1) + ['C'] + random.sample(AAs,m)+['C']+random.sample(AAs,n)+['C']+random.sample(AAs,o) + ['C'] + random.sample(AAs,1))]))
    return peptides

def getPeptideFramework(m,n,o,num_prolines,AAs=aa_set) -> list:
    '''
    getPeptideFramework: return cyclic peptide inner amino acids like Kong2020
    Inputs:
        m: int, None: random
        n: int, None: random
        o: int, None: random
        num_prolines: number of prolines to sample
        AAs: set of amino acids to be used
    Outputs:
        randomized peptide sequence(s)
    '''
    #prevent generation of extra prolines
    try:
        AAs.remove('P')
    except KeyError:
        pass
    except AttributeError:
        print('wrong data type for AAs. needs to be a set')
        
    rand=False
    if (m is None) and (n is None) and (o is None):
        rand=True
    elif (m is None) or (n is None) or (o is None):
        print('need to specify all three or zero')
        return None

    peptides = []
    if rand: #if no (m,n,o) given, assign it
        total_length = random.randint(3,8)
        m = random.randint(1,total_length-2)
        n = random.randint(1,total_length-1-m)
        o = random.randint(1,total_length-m-n)
        x = [m,n,o]
        random.shuffle(x)
        m,n,o = [i for i in x]

    #get permutations of proline placements
    combinations = ['X']*(m+n+o-num_prolines)+['P']*num_prolines
    combinations = list(set(itertools.permutations(combinations))) #["".join([str(i) for i in string]) for string in list(set(itertools.permutations(combinations)))]
    combinations = [list(i) for i in combinations]

    #get AAs to fill X's
    inter_AAs = random.sample(AAs,m+n+o)
    random.shuffle(inter_AAs)

    peptides=[]
    #replace X with random AA's, skipping prolines
    for c in range(len(combinations)):
        j=0
        for i in range(m+n+o):
            if combinations[c][i] != 'P':
                combinations[c][i] = inter_AAs[j]
                j+=1
    combinations = ["".join([str(i) for i in c]) for c in combinations]
    #add to list
    [peptides.append(c) for c in combinations]
    return peptides

def getFullPeptide(m,n,o,inter_AA,AAs=aa_set):
    '''
    getFullPeptide: return full amino acid cyclic peptide like Kong2020
    Inputs:
        m: int, None: random
        n: int, None: random
        o: int, None: random
        inter_AA: single sequence, output from getPeptide Framework()
        AAs: set of amino acids to be used
    Outputs:
        full peptide sequence
    '''
    sequence = "".join([str(i) for i in (random.sample(AAs,1) + ['C'] + [inter_AA[:m]]+['C']+[inter_AA[m:m+n]]+['C']+[inter_AA[m+n:m+n+o]] + ['C'] + random.sample(AAs,1))])
    return sequence

