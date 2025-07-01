import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.momamba.MoMBlock import MoMBlock

class MoMCodec(nn.Module):
    def __init__(self,in_channels, num_experts,top_k):
        super().__init__()

        #stage blocks
        self.mom = MoMBlock(in_channels,num_experts, top_k,
                            emb_type="PE",
                            head=2,
                            use_aux_loss=False)

    def forward(self, x):  
        aux_loss=0  
        x,aux_loss = self.mom(x)
        return x,aux_loss

class StageEncoder(nn.Module):
    def __init__(self,in_channels, out_channels, num_experts,top_k):
        super().__init__()

        #stage blocks
        self.encoder = MoMCodec(out_channels, num_experts,top_k)
        

    def forward(self, x):
        x,  aux_loss= self.encoder(x)
        return x, aux_loss

class MoMamba_Dn(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 config,  
                 #nc=5,
                 nc=6, 
                 num_of_layers=17) : 
        super(MoMamba_Dn, self).__init__()

        kernel_size = 3
        padding = 1
        self.layers = []
        
        conv1 = nn.Conv3d(in_channels=in_channels, out_channels=nc, kernel_size=kernel_size, padding=padding, bias=False)
        conv1.weight.data.fill_(1/9)
        self.layers.append(conv1)
        self.layers.append(nn.ReLU(inplace=True))
        
        top_k = 2
        num_expert = 3

        for i in range(num_of_layers-2):
            if (i != 9) :
                self.layers.append(nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=kernel_size, padding=padding, bias=False))
                self.layers.append(nn.ReLU(inplace=True))
            else:
                self.layers.append(StageEncoder(nc,nc,num_expert, top_k))
        
        conv_end = nn.Conv3d(in_channels=nc, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False)

        self.layers.append(conv_end)
        self.dncnn3d = nn.Sequential(*self.layers)
        self.return_aux_loss=True
        
        
    def forward(self, x):
        #out = self.dncnn3d(x)
        input = x# x.detach()
        all_aux_loss = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, StageEncoder):
                x, aux_loss =layer(x)
                all_aux_loss+=aux_loss
            else:
                x =layer(x)

        #add an relu layer to keep the result image to be positive
        res = F.relu(input - x)
    
        if self.return_aux_loss:
            return [res, all_aux_loss]
        else:
            return res
