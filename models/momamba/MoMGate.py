import torch
from torch import Tensor, nn

class MultiHeadRouting(nn.Module):
    def __init__(
        self,
        dim,
        num_experts,
        head,
        drop=0.2
    ):
        super().__init__()  
        self.head=head

        self.w_gate = nn.Linear(dim, num_experts*head)
        if self.head > 1:
            self.act=nn.GELU()                   
            self.gate_proj = nn.Linear(num_experts*head, num_experts)
            self.drop=nn.Dropout(drop)

    def forward(self, x: Tensor):
        logits=self.w_gate(x)

        if self.head>1: 
            logits=self.act(logits)           
            logits=self.gate_proj(logits)
            logits=self.drop(logits)
        
        return logits



class MoMGate(nn.Module):
    def __init__(
        self,
        dim:int,
        num_experts: int,
        top_k:int,
        head,
        use_aux_loss
    ):
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.top_k=top_k

        self.mhr=MultiHeadRouting(dim,num_experts,head)

        self.use_aux_loss =use_aux_loss

        self.capacity_factor = 1.0
        self.epsilon = 1e-6 

        ratio=0.1
        self.balance_loss_coef = (1e-2)*ratio
        self.router_z_loss_coef = (1e-3)*ratio

    def forward(self, x: Tensor):       
        # Compute gate scores
        logits=self.mhr(x)

        gate_scores = nn.functional.softmax(logits, dim=-1)
        top_k_scores, top_k_indices = gate_scores.topk(self.top_k, dim=-1)
  
        #if self.top_k>1:
        #    top_k_scores = top_k_scores / (top_k_scores.sum(-1, keepdim=True) + 1e-6)  # normalization
 
        routed_experts = torch.zeros_like(gate_scores).scatter_(
            dim=-1,
            index=top_k_indices,
            src=torch.ones_like(top_k_scores),
        )

        # Compute loss
        aux_loss = 0
        if self.training and self.use_aux_loss:
            if self.balance_loss_coef>0:
                total_tokens = x.shape[0] * x.shape[1]
                f_i = torch.sum(routed_experts, dim=(0, 1)) * (1 / total_tokens)
                P_i = (torch.sum(gate_scores, dim=(0, 1))) * (1 / total_tokens)
                aux_loss += self.balance_loss_coef * self.num_experts * torch.sum((f_i * P_i))

            if self.router_z_loss_coef>0:
                router_z_loss = torch.logsumexp(gate_scores, dim = -1)
                router_z_loss = torch.square(router_z_loss)            
                router_z_loss = router_z_loss.mean()
                aux_loss+=self.router_z_loss_coef*router_z_loss

        # Capacity
        '''
        capacity = ((x.shape[0] * x.shape[1]) / self.num_experts) * self.capacity_factor        
        flat_routed_experts = routed_experts.view(-1, self.num_experts)
        total_expert_allocation = torch.cumsum(flat_routed_experts, dim=0)
        expert_mask = (total_expert_allocation <= capacity).float()
        revised_expert_allocation = expert_mask * flat_routed_experts
        routed_experts = revised_expert_allocation.view(
            routed_experts.shape
        )
        '''
        #zeros = torch.zeros_like(gate_scores, requires_grad=True)
        #routed_expert_probs = zeros.scatter(-1, top_k_indices, top_k_scores)

        #routed_expert_probs = gate_scores * routed_experts

        #return top_k_scores, top_k_indices, aux_loss
        return gate_scores, routed_experts, aux_loss