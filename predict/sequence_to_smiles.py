# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:19:48 2022

@author: marsh
"""
import sys
import pandas as pd
import numpy as np
import re

# =============================================================================
# How to use:
# =============================================================================
#First, ensure this python script and the xlsx dictionary are in the same folder
#Then, run this program (F5)
#Then, in the console, type (your peptide sequence should have Z in place of AHA):
    #AA = '[peptide sequence goes here]'
#If you want no C or N cap, type:
    #smiles_seq = getSmilesFromAA(AA)
#If you want C and N cap, type:
    #smiles_seq = getSmilesFromAA(AA,N_term='acetyl',C_term='amine')
#Finally, type:
    #smiles_seq = getStapledSeqFromAHA(smiles_seq)
    
# =============================================================================
# Code
# =============================================================================

#import smiles strings of every amino acid
aa_df = pd.read_excel('AA_smiles_dict.xlsx')
aa_df = aa_df.set_index('1 letter')
aa_dict = dict(zip(aa_df.index,aa_df['full smiles']))

#PE smiles format:
propargyl_ether = 'N1N=NC(=C1)C[O]CC1=CN(N=N1)'
linker_7 = 'CC(Cn)=O' #Kong et al. Nature biomedical engineering 2020

if len(sys.argv) > 1:
    input_seq = sys.argv[1]
else:
    input_seq = 'none'

def getSmilesFromAA(AA_seq,N_term='',C_term=''):
    #AA_seq: sequence of amino acids with stanard 20 + 'Z' for azidohomoalanine
    #N_term: acetyl for CC(=O) cap
    #C_term: amine for NH2 cap
    if AA_seq.count('X') > 0:
        AA_seq = AA_seq.replace('X','Z')
    if C_term == 'amine':
        c_cap = 'N'
    else:
        c_cap = 'O'
    if N_term == 'acetyl':
        n_cap = 'CC(=O)'
    else:
        n_cap = ''
    return n_cap+''.join([aa_dict[c] for c in AA_seq])+c_cap

def formDisulfide(smiles_seq):
    #turn a peptide with two disulfides into a disulfide bond
    return smiles_seq.replace('S','S3')

def getStapledSeqFromDisulfide(smiles_seq,staple):
    #make a stapled peptide by replacing a disulfide. not recommended
    if smiles_seq.find('SS')!= -1:
        print(smiles_seq.replace('SS',staple))
        return smiles_seq.replace('SS',staple)
    else:
        print('Could not find disulfide bond to replace')
    
def getStapledSeqFromAHA(smiles_seq,staple='N1N=NC(=C1)C[O]CC1=CN(N=N1)'):
    #replace two aha residues with a PE staple. recommended
    aha_len = 11
    if smiles_seq.count('N=[N+]=[N-]')!=2:
        print('incorrect number of AHA residues, needs to have two')
    elif smiles_seq.count('N=[N+]=[N-]')==2:
        aha1=smiles_seq.find('N=[N+]=[N-]')
        smiles_seq_updated = smiles_seq[:aha1]+staple+'3'+smiles_seq[aha1+11:]
        aha2=smiles_seq_updated.find('N=[N+]=[N-]')
        smiles_seq_updated=smiles_seq_updated[:aha2]+'3'+smiles_seq_updated[aha2+11:]
        return(smiles_seq_updated)
    
def getCrosslinkedSeq(smiles_seq,staple):
    #get 3 reacted products from cyclic peptide reaction from Wong et al. 2020
    if smiles_seq.count('S)') != 4:
        print('incorrect number of Cys residues, needs to have four')
        return None
    elif len(staple.split('n')) != 2:
        print('incorrect attachment form of the linker. Linker needs to like: CC(Cn)=O where \'n\' is the attachment point for the second atom')
        return None
    else:
        c1,c2,c3,c4=[m.start() for m in re.finditer('S\)',smiles_seq)]
        smiles_seq_updated_1 = smiles_seq[:c1+1]+staple.split('n')[0] + str(3) + staple.split('n')[1]+smiles_seq[c1+1:c2+1]+'3'+\
            smiles_seq[c2+1:c3+1] + staple.split('n')[0] + str(3) + staple.split('n')[1] + smiles_seq[c3+1:c4+1] + '3' +\
            smiles_seq[c4+1:]
        smiles_seq_updated_2 = smiles_seq[:c1+1]+staple.split('n')[0] + str(4) + staple.split('n')[1]+smiles_seq[c1+1:c2+1]+'3'+\
            smiles_seq[c2+1:c3+1] + staple.split('n')[0] + str(3) + staple.split('n')[1] + smiles_seq[c3+1:c4+1] + '4' +\
            smiles_seq[c4+1:]
        smiles_seq_updated_3 = smiles_seq[:c1+1]+staple.split('n')[0] + str(3) + staple.split('n')[1] + smiles_seq[c1+1:c2+1] +\
            staple.split('n')[0] + '4'+ staple.split('n')[1]+ smiles_seq[c2+1:c3+1]  + str(3)  + smiles_seq[c3+1:c4+1] + '4' +\
            smiles_seq[c4+1:]
        return smiles_seq_updated_1,smiles_seq_updated_2,smiles_seq_updated_3