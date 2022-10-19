import torch
import torch.nn as nn
import rdkit.Chem as Chem

class HierPredict(nn.Module):
    def __init__(self, args):
        super(HierPredict, self).__init__()
        self.ff1 = nn.Sequential( 
                nn.Linear(args.latent_size, args.latent_size), 
                nn.ReLU(),
                nn.Dropout(args.dropout)
        )
        
        self.ff2 = nn.Sequential( 
                nn.Linear(args.latent_size, args.latent_size), 
                nn.ReLU(),
                nn.Dropout(args.dropout)
        )
        
        self.ff3 = nn.Sequential(nn.Linear(args.latent_size,args.label_size),
                # nn.Softmax(dim=1)         
        )
        
    def forward(self,x):
        out = self.ff1(x)
        out = self.ff2(out)
        out = self.ff3(out)
        return out