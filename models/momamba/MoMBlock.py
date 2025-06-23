import torch
from torch import Tensor, nn
from torch import nn, einsum
from mamba_ssm import Mamba
from models.momamba.UNetBlock import UnetResBlock, get_conv_layer,PositionalEncoding,RotaryPositionalEmbeddings,LearnedPositionEmbeddings
from models.momamba.MoMGate import *



class MoMBlock(nn.Module):
    def __init__(self, dim, num_experts,top_k,emb_type,head,use_aux_loss):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim), requires_grad=True)

        d_state = 16
        d_conv = 4 
        expand = 2

        if num_experts==1:
            self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
            )
        else:
            self.mamba = MoMEncoder(
                d_model=dim, # Model dimension d_model
                num_experts=num_experts,
                top_k=top_k,
                emb_type=emb_type,
                head=head,
                use_aux_loss=use_aux_loss
            )
    
    #@autocast(enabled=False)
    def forward(self, x):
        B, C, H, W, D = x.shape
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)
        x_norm =self.norm(x)
        
        aux_loss=0
        if isinstance(self.mamba,(MoMEncoder)):
            x_mamba, aux_loss =self.mamba(x_norm)
        else:
            x_mamba = self.mamba(x_norm)

        x_mamba =  x+self.gamma* x_mamba
        x = x_mamba.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        return x, aux_loss


class MoMEncoder(nn.Module):
    def __init__(self, d_model, num_experts,top_k, emb_type,head,use_aux_loss):
        super().__init__()

        self.emb_type=emb_type
        if self.emb_type=="PE":
            self.pos_emb=PositionalEncoding(d_model)
        elif self.emb_type=="RPE":
            self.pos_emb=RotaryPositionalEmbeddings(d_model)
        elif self.emb_type=="LE":#RoBERTa,BERT、GPT
            self.pos_emb=LearnedPositionEmbeddings(d_model)

        # instantiate experts
        d_state = 16
        d_conv = 4 
        expand = 2
        
        self.experts = [Mamba(
                d_model=d_model, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
            ) for i in range(num_experts)]
        self.experts = nn.ModuleList(self.experts)

        self.top_k=top_k
        self.gate = MoMGate(d_model, num_experts,top_k,head, use_aux_loss) 

    def forward(self, x):
        if self.emb_type !="":
            x=self.pos_emb(x)

        gate_scores, routed_experts, aux_loss = self.gate(x)

        output = torch.zeros_like(x)
        batch=x.shape[0]
        for b in range(batch):
            for i, expert in enumerate(self.experts):
                mask = routed_experts[b,:,i] == 1
                if mask.any():
                    expert_x = x[b,mask,:]
                    expert_x=expert_x.view(1,expert_x.shape[0],expert_x.shape[1])
                    expert_output = expert(expert_x)
                    expert_output = expert_output.view(-1,expert_x.shape[2])
                    probs=gate_scores[b,mask,i].view(-1,1)
                    expert_output =expert_output*probs
                    output[b,mask] += expert_output

        '''
        #expert_probs, expert_indices, aux_loss = self.gate(x)
        output = torch.zeros_like(x)
        sep_batch=True       
        if sep_batch:
             #seperate batch
            batch=x.shape[0]
            for b in range(batch):
                for k in range(self.top_k):
                    for i, expert in enumerate(self.experts):
                        mask = expert_indices[b,:,k] == i
                        if mask.any():
                            expert_x = x[b,mask,:]
                            expert_x=expert_x.view(1,expert_x.shape[0],expert_x.shape[1])
                            expert_output = expert(expert_x)
                            expert_output = expert_output.view(-1,expert_x.shape[2])
                            probs=expert_probs[b,mask,k].view(-1,1)
                            expert_output =expert_output*probs
                            output[b,mask] += expert_output
        else:
            #pack batchs
            expert_probs, expert_indices = expert_probs.view(-1, self.top_k), expert_indices.view(-1, self.top_k)
            x_flat=x.view(-1,x.shape[2])
            out_flat= torch.zeros_like(x_flat)
            for k in range(self.top_k):
                for i, expert in enumerate(self.experts):
                    mask = expert_indices[:,k] == i
                    if mask.any():
                        expert_x = x_flat[mask,:]
                        expert_x=expert_x.view(1,expert_x.shape[0],expert_x.shape[1])
                        expert_output = expert(expert_x)
                        expert_output = expert_output.view(-1,expert_x.shape[2])
                        probs=expert_probs[mask,k].view(-1,1)
                        expert_output =expert_output*probs
                        out_flat[mask] += expert_output
            output=out_flat.view(x.shape)
        '''    
        return output, aux_loss