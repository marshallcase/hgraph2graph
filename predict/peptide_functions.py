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
import rdkit

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

def mol2seq(m):
    '''
    mol2seq: convert generated peptide smiles to amino acid sequence, otherwise return False
    adapted from https://github.com/rdkit/rdkit/discussions/4659
    Inputs:
        m (Chem.mol - RDKit)
    Outputs:
        Amino acid sequence (str) otherwise False
    '''
    aa_smiles = {'ALA': 'C[C@H](N)C=O', 'CYS': 'N[C@H](C=O)CS', 'ASP': 'N[C@H](C=O)CC(=O)O', 'GLU': 'N[C@H](C=O)CCC(=O)O', 'PHE': 'N[C@H](C=O)Cc1ccccc1', 'GLY': 'NCC=O', 'HIS': 'N[C@H](C=O)Cc1c[nH]cn1', 'ILE': 'CC[C@H](C)[C@H](N)C=O', 'LYS': 'NCCCC[C@H](N)C=O', 'LEU': 'CC(C)C[C@H](N)C=O', 'MET': 'CSCC[C@H](N)C=O', 'ASN': 'NC(=O)C[C@H](N)C=O', 'PRO': 'O=C[C@@H]1CCCN1', 'GLN': 'NC(=O)CC[C@H](N)C=O', 'ARG': 'N=C(N)NCCC[C@H](N)C=O', 'SER': 'N[C@H](C=O)CO', 'THR': 'C[C@@H](O)[C@H](N)C=O', 'VAL': 'CC(C)[C@H](N)C=O', 'TRP': 'N[C@H](C=O)Cc1c[nH]c2ccccc12','TYR': 'N[C@H](C=O)Cc1ccc(O)cc1','STP':'FC[C@@H](C=O)N'}
    aas = ['GLY','ALA', 'VAL', 'CYS', 'ASP', 'GLU', 'PHE', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'TRP','TYR','STP'] #order important because gly is substructure of other aas
    #replace cyclic peptide residue with 'STP' residue
    m = (Chem.ReplaceSubstructs(m, 
                                 Chem.MolFromSmiles('CSCC(CSC)=O'), 
                                 Chem.MolFromSmiles('CF'),
                                 replaceAll=True)[0])
    # detect the atoms of the backbone and assign them with info
    CAatoms = m.GetSubstructMatches(Chem.MolFromSmarts("[C:0](=[O:1])[C:2][N:3]"))
    for atoms in CAatoms:
        a = m.GetAtomWithIdx(atoms[2])
        info = Chem.AtomPDBResidueInfo()
        info.SetName(" CA ") #spaces are important
        a.SetMonomerInfo(info)
    # detect the presence of residues and set residue name for CA atoms only
    for curr_aa in aas:
        matches = m.GetSubstructMatches(Chem.MolFromSmiles(aa_smiles[curr_aa]))
        for atoms in matches:
            for atom in atoms:
                a = m.GetAtomWithIdx(atom)
                info = Chem.AtomPDBResidueInfo()
                if a.GetMonomerInfo() != None:
                    if a.GetMonomerInfo().GetName() == " CA ":
                        info.SetName(" CA ")
                        info.SetResidueName(curr_aa)
                        a.SetMonomerInfo(info)
    # renumber the backbone atoms so the sequence order is correct:
    bbsmiles = "O"+"C(=O)CN"*len(m.GetSubstructMatches(Chem.MolFromSmiles(aa_smiles["GLY"]))) # generate backbone SMILES
    backbone = m.GetSubstructMatches(Chem.MolFromSmiles(bbsmiles))[0]
    id_list = list(backbone)
    id_list.reverse()
    for idx in [a.GetIdx() for a in m.GetAtoms()]:
        if idx not in id_list:
            id_list.append(idx)
    m_renum = Chem.RenumberAtoms(m,newOrder=id_list)
    return Chem.MolToSequence(m_renum)

def getUniqueness(smiles_list):
    '''
    getUniqueness: get fraction of unique molecules in a list of smiles
    Inputs:
        smiles_list: list of smiles
    Outputs:
        fraction_unique: fraction of unique molecules (float between 0 and 1)
    '''
    mols = pd.DataFrame(smiles_list)
    unique_mols = mols[mols.columns[0]].unique()
    fraction_unique = len(unique_mols)/len(mols)
    return fraction_unique

def upper_tri_indexing(A): #https://stackoverflow.com/questions/47314754/how-to-get-triangle-upper-matrix-without-the-diagonal-using-numpy
    '''
    upper_tri_indexing: get upper diagonal elements of numpy array without diagonal elements
    Inputs:
        A: numpy array
    Outputs:
        1D numpy array of elements
    '''
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]

def getSimilarityMatrix(smiles_list):
    '''
    getSimilarityMatrix: get matrix of molecule similarity
    Inputs:
        smiles_list: list of smiles
    Outputs:
        similarities: numpy array of Tanimoto similarities between peptides
    '''
    ms = [Chem.MolFromSmiles(i) for i in smiles_list]
    fps = [Chem.RDKFingerprint(x) for x in ms]
    similarities = np.zeros((len(smiles_list),len(smiles_list)))
    for i,m1 in enumerate(fps):
        for j,m2 in enumerate(fps[i:]):
            similarities[i,j+i] = DataStructs.FingerprintSimilarity(m1,m2,metric=DataStructs.TanimotoSimilarity)
    return similarities

def mol2seq(m):
    '''
    mol2seq: checks if a smiles string is a valid peptide
    Inputs:
        m: smiles string, peptide-like
    Outputs:
        True or False
    '''
    try:
        m=Chem.ReplaceSubstructs(Chem.MolFromSmiles(m), 
                                     Chem.MolFromSmiles('CSCC(CSC)=O'), 
                                     Chem.MolFromSmiles('F'),
                                     replaceAll=True)[0]
        aa_smiles = {'ALA': 'C[C@H](N)C=O', 'CYS': 'N[C@H](C=O)CS', 'ASP': 'N[C@H](C=O)CC(=O)O', 'GLU': 'N[C@H](C=O)CCC(=O)O', 'PHE': 'N[C@H](C=O)Cc1ccccc1', 'GLY': 'NCC=O', 'HIS': 'N[C@H](C=O)Cc1c[nH]cn1', 'ILE': 'CC[C@H](C)[C@H](N)C=O', 'LYS': 'NCCCC[C@H](N)C=O', 'LEU': 'CC(C)C[C@H](N)C=O', 'MET': 'CSCC[C@H](N)C=O', 'ASN': 'NC(=O)C[C@H](N)C=O', 'PRO': 'O=C[C@@H]1CCCN1', 'GLN': 'NC(=O)CC[C@H](N)C=O', 'ARG': 'N=C(N)NCCC[C@H](N)C=O', 'SER': 'N[C@H](C=O)CO', 'THR': 'C[C@@H](O)[C@H](N)C=O', 'VAL': 'CC(C)[C@H](N)C=O', 'TRP': 'N[C@H](C=O)Cc1c[nH]c2ccccc12','TYR': 'N[C@H](C=O)Cc1ccc(O)cc1'}
        aas = ['GLY','ALA', 'VAL', 'CYS', 'ASP', 'GLU', 'PHE', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'TRP','TYR'] #order important because gly is substructure of other aas
        # detect the atoms of the backbone and assign them with info
        CAatoms = m.GetSubstructMatches(Chem.MolFromSmarts("[C:0](=[O:1])[C:2][N:3]"))
        for atoms in CAatoms:
            a = m.GetAtomWithIdx(atoms[2])
            info = Chem.AtomPDBResidueInfo()
            info.SetName(" CA ") #spaces are important
            a.SetMonomerInfo(info)
        # detect the presence of residues and set residue name for CA atoms only
        for curr_aa in aas:
            matches = m.GetSubstructMatches(Chem.MolFromSmiles(aa_smiles[curr_aa]))
            for atoms in matches:
                for atom in atoms:
                    a = m.GetAtomWithIdx(atom)
                    info = Chem.AtomPDBResidueInfo()
                    if a.GetMonomerInfo() != None:
                        if a.GetMonomerInfo().GetName() == " CA ":
                            info.SetName(" CA ")
                            info.SetResidueName(curr_aa)
                            a.SetMonomerInfo(info)
        # renumber the backbone atoms so the sequence order is correct:
        bbsmiles = "O"+"C(=O)CN"*len(m.GetSubstructMatches(Chem.MolFromSmiles(aa_smiles["GLY"]))) # generate backbone SMILES
        backbone = m.GetSubstructMatches(Chem.MolFromSmiles(bbsmiles))[0]
        id_list = list(backbone)
        id_list.reverse()
        for idx in [a.GetIdx() for a in m.GetAtoms()]:
            if idx not in id_list:
                id_list.append(idx)
        m_renum = Chem.RenumberAtoms(m,newOrder=id_list)
        #output sequence
        # Chem.MolToSequence(m_renum)
    except:
        return False
    return True 