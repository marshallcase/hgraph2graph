import rdkit
import rdkit.Chem as Chem
import copy
import torch

class Vocab(object):

    def __init__(self, smiles_list):
        self.vocab = [x for x in smiles_list] #copy
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        
    def __getitem__(self, smiles):
        return self.vmap[smiles]

    def get_smiles(self, idx):
        return self.vocab[idx]

    def size(self):
        return len(self.vocab)

class PairVocab(object):

    def __init__(self, smiles_pairs, cuda=True):
        cls = list(zip(*smiles_pairs))[0]
        self.hvocab = sorted( list(set(cls)) )
        self.hmap = {x:i for i,x in enumerate(self.hvocab)}

        self.vocab = [tuple(x) for x in smiles_pairs] #copy
        self.inter_size = [count_inters(x[1]) for x in self.vocab]
        self.vmap = {x:i for i,x in enumerate(self.vocab)}

        self.mask = torch.zeros(len(self.hvocab), len(self.vocab))
        for h,s in smiles_pairs:
            hid = self.hmap[h]
            idx = self.vmap[(h,s)]
            self.mask[hid, idx] = 1000.0

        if cuda: self.mask = self.mask.cuda()
        self.mask = self.mask - 1000.0
        
        self.backbone_vocab = ['CN1CCCC1[CH:1]=O','O=CC1CCCN1[CH3:1]','CNCC=O',
                               'CNC[CH:1]=O','CN[CH2:1]C=O','O=CCN[CH3:1]']
        self.c_terminus_vocab = ['O=C(O)C1CCCN1[CH3:1]','O=C(O)CN[CH3:1]']
        self.n_terminus_vocab = ['NC[CH:1]=O','O=[CH:1]C1CCCN1']
        self.staple_vocab = ['CCSCC(=O)CSC[CH3:1]']
        
        self.backbone_hvocab = ['CNCC=O','CN1CCCC1C=O']
        self.c_terminus_hvocab = ['CN1CCCC1C(=O)O','CNCC(=O)O']
        self.n_terminus_hvocab = ['NCC=O','O=CC1CCCN1']
        self.staple_hvocab = ['CCSCC(=O)CSCC']
                               
            
    def __getitem__(self, x):
        assert type(x) is tuple
        return self.hmap[x[0]], self.vmap[x]

    def get_smiles(self, idx):
        return self.hvocab[idx]

    def get_ismiles(self, idx):
        return self.vocab[idx][1] 

    def size(self):
        return len(self.hvocab), len(self.vocab)

    def get_mask(self, cls_idx):
        return self.mask.index_select(index=cls_idx, dim=0)

    def get_inter_size(self, icls_idx):
        return self.inter_size[icls_idx]
    
    def is_backbone_vocab(self,vocab_unit):
        return vocab_unit in self.backbone_vocab
    
    def is_backbone_hvocab(self,hvocab_unit):
        return hvocab_unit in self.backbone_hvocab
    
    def is_c_terminus_vocab(self,vocab_unit):
        return vocab_unit in self.c_terminus_vocab
    
    def is_c_terminus_hvocab(self,hvocab_unit):
        return hvocab_unit in self.c_terminus_hvocab
    
    def is_n_terminus_vocab(self,vocab_unit):
        return vocab_unit in self.n_terminus_vocab
    
    def is_n_terminus_hvocab(self,hvocab_unit):
        return hvocab_unit in self.n_terminus_hvocab
    
    def is_staple_vocab(self,vocab_unit):
        return vocab_unit in self.staple_vocab
    
    def is_staple_hvocab(self,hvocab_unit):
        return hvocab_unit in self.staple_hvocab
    

COMMON_ATOMS = [('B', 0), ('B', -1), ('Br', 0), ('Br', -1), ('Br', 2), ('C', 0), ('C', 1), ('C', -1), ('Cl', 0), ('Cl', 1), ('Cl', -1), ('Cl', 2), ('Cl', 3), ('F', 0), ('F', 1), ('F', -1), ('I', -1), ('I', 0), ('I', 1), ('I', 2), ('I', 3), ('N', 0), ('N', 1), ('N', -1), ('O', 0), ('O', 1), ('O', -1), ('P', 0), ('P', 1), ('P', -1), ('S', 0), ('S', 1), ('S', -1), ('Se', 0), ('Se', 1), ('Se', -1), ('Si', 0), ('Si', -1)]
common_atom_vocab = Vocab(COMMON_ATOMS)

def count_inters(s):
    mol = Chem.MolFromSmiles(s)
    inters = [a for a in mol.GetAtoms() if a.GetAtomMapNum() > 0]
    return max(1, len(inters))

def flatten(l):
    return [item for sublist in l for item in sublist]
