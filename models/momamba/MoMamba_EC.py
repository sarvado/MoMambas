import torch
from urllib.parse import DefragResult
from torch import nn
from typing import Tuple, Union
from models.ec_unetr.dynunet_block import UnetOutBlock, UnetResBlock

import numpy as np

from base64 import encode
from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from models.ec_unetr.layers import LayerNorm
from models.ec_unetr.dynunet_block import get_conv_layer, UnetResBlock

from models.momamba.MoMBlock import MoMBlock

class MoMCodec(nn.Module):
    def __init__(self, dim, num_experts,mom_config):
        super().__init__()

        top_k=mom_config["top_k"]
        emb_type=mom_config["emb_type"]
        head=mom_config["head"]
        use_aux_loss=mom_config["use_aux_loss"]

        self.mom = MoMBlock(dim, num_experts, top_k,
                            emb_type=emb_type,
                            head=head,
                            use_aux_loss=use_aux_loss)

        self.conv51 = UnetResBlock(3, dim, dim, kernel_size=3, stride=1, norm_name="batch")
        self.conv52 = UnetResBlock(3, dim, dim, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(dim, dim, 1))

    def forward(self, x):
        x, aux_loss=self.mom(x)

        res = self.conv51(x)
        res = self.conv52(res)
        x = x + self.conv8(res)
        return x, aux_loss


class StageEncoder(nn.Module):
    def __init__(self, in_channels,out_channels,num_experts,mom_config,
                  depth=3,  spatial_dims=3, dropout=0.0 ):
        super().__init__()

        #downsample: conv+norm
        self.downsample_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=out_channels),
        )

        #stage blocks
        stage_blocks = []          
        for j in range(depth):
            stage_blocks.append(MoMCodec(out_channels,num_experts[j],mom_config))  
        self.stage_blocks=nn.ModuleList(stage_blocks)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.downsample_layer(x)
        all_aux_loss=0
        for j in range(len(self.stage_blocks)):
            x, aux_loss = self.stage_blocks[j](x)
            all_aux_loss += aux_loss
        return x, all_aux_loss

class CrossOutBlock(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, upsample_stride:int=2
    ):
        super().__init__()
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_stride,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

    def forward(self, inp):
        return self.transp_conv(inp)

class StageDecoder(nn.Module):
    def     __init__(
            self,            
            in_channels: int,
            out_channels: int,
            num:int=2,
            isconv:bool=True,
            num_experts=1,
            mom_config=None,
            spatial_dims: int=3,          
            depth: int = 3,
            kernel_size: Union[Sequence[int], int]=3,
            upsample_kernel_size: Union[Sequence[int], int]=2,
            norm_name: Union[Tuple, str]= "instance",
    ) -> None:

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        #sum
        middle_channels=out_channels#*num
        #cat
        #middle_channels=out_channels*num
        self.alpha = nn.Parameter(torch.ones(num,1,middle_channels,1,1,1))
        self.gamma = nn.Parameter(torch.zeros(num,1,middle_channels,1,1,1))
        self.beta = nn.Parameter(torch.zeros(num,1,middle_channels,1,1,1))
        self.epsilon = 1e-5

        self.catconv = nn.Sequential(nn.Conv3d(middle_channels, out_channels, kernel_size, stride=1, padding=1),
                                     nn.BatchNorm3d(out_channels),
                                     nn.ReLU(inplace=True), ) 

        if isconv:
            self.decoder_block=UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                                norm_name=norm_name, )
        else:
            decoder_block = []
            for j in range(depth):
                decoder_block.append(MoMCodec(out_channels,num_experts[j],mom_config))
            self.decoder_block=nn.ModuleList(decoder_block)        
        self.apply(self._init_weights) 

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    #sum
    def forward(self, *inp):     
        inp = list(inp) 
        inp[0] = self.transp_conv(inp[0])
        inp = torch.stack(inp)

        all_aux_loss=0
        if isinstance(self.decoder_block,(UnetResBlock)):
            embedding = (inp.pow(2).sum((2,3,4,5), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)            
            gate = 1. + torch.tanh(embedding * norm + self.beta)               
            inp = gate * inp

            out=inp.sum(0)
            out = self.decoder_block(out)
        else:
            out=inp.sum(0)
            for j in range(len(self.decoder_block)):
                out, aux_loss = self.decoder_block[j](out)
                all_aux_loss += aux_loss
        return out, all_aux_loss
    '''
    
    #cat
    def forward(self, *inp):       
        out = self.transp_conv(inp[0])

        for i in range(1,len(inp)):
            out = torch.cat([out, inp[i]], 1)
        
        if isinstance(self.decoder_block,(UnetResBlock)):
            embedding = (out.pow(2).sum((2,3,4), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)            
            gate = 1. + torch.tanh(embedding * norm + self.beta)               
            out = gate * out

        out=self.catconv(out)

        out = self.decoder_block(out)
        return out
    '''

class MoMamba_EC(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            config,
            feature_size: int = 8,
            num_heads: int = 4,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=[3,3,3,3],
            dims=[16,32,64,128],
            conv_op=nn.Conv3d,

    ) -> None:
        super().__init__()
        
        mom_config=config["mom"]
        num_experts=[[3,1,1],[3,1,1],[3,1,1],[3,1,1]]
        for stage_id in range(len(mom_config["num_experts"])):
            num_experts[stage_id][0]=mom_config["num_experts"][stage_id]

        dims.insert(0,feature_size)

        self.conv00 = UnetResBlock(spatial_dims=3,in_channels=in_channels,out_channels=feature_size,
                                   kernel_size=3,stride=1,norm_name=norm_name)
        self.conv10 = StageEncoder(in_channels, dims[1], num_experts[0], mom_config)
        self.conv20 = StageEncoder(dims[1], dims[2], num_experts[1], mom_config)
        self.conv30 = StageEncoder(dims[2], dims[3], num_experts[2], mom_config)
        self.conv40 = StageEncoder(dims[3], dims[4], num_experts[3], mom_config)

        # upsampling
        self.up_concat01 = StageDecoder(dims[1], dims[0])
        self.up_concat11 = StageDecoder(dims[2], dims[1])
        self.up_concat21 = StageDecoder(dims[3], dims[2])
        self.up_concat31 = StageDecoder(dims[4], dims[3],2,False, num_experts[2], mom_config)

        self.up_concat02 = StageDecoder(dims[1], dims[0],3)
        self.up_concat12 = StageDecoder(dims[2], dims[1],3)
        self.up_concat22 = StageDecoder(dims[3], dims[2],3,False, num_experts[1], mom_config)

        self.up_concat03 = StageDecoder(dims[1], dims[0],4)
        self.up_concat13 = StageDecoder(dims[2], dims[1],4,False, num_experts[0], mom_config)

        self.up_concat04 = StageDecoder(dims[1], dims[0],5)

        # final conv (without any concat)
        self.out_1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        self.out_2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        self.out_3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        self.out_4 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)

        self.out_5 = CrossOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels, upsample_stride=2)
        self.out_6 = CrossOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels, upsample_stride=4)
        self.out_7 = CrossOutBlock(spatial_dims=3, in_channels=feature_size * 8, out_channels=out_channels, upsample_stride=8)
        self.out_8 = CrossOutBlock(spatial_dims=3, in_channels=feature_size * 16, out_channels=out_channels, upsample_stride=16)

        self.weights = nn.Parameter(torch.ones(8,1,1,1,1,1)*(1.0/8))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_in):
        aux_loss=np.zeros(14)
        # column : 0
        X_00 = self.conv00(x_in)#8*(48)
        T_10, aux_loss[0] = self.conv10(x_in)#16*(24)
        T_20, aux_loss[1] = self.conv20(T_10)#32*(12)
        T_30, aux_loss[2] = self.conv30(T_20)#64*(6)
        T_40, aux_loss[3] = self.conv40(T_30)#128*(3)

        # column : 1
        X_01, aux_loss[4] = self.up_concat01(T_10, X_00)#8*(48)
        X_11, aux_loss[5] = self.up_concat11(T_20, T_10)#16*(24)
        X_21, aux_loss[6] = self.up_concat21(T_30, T_20)#32*(12)
        X_31, aux_loss[7] = self.up_concat31(T_40, T_30)#64*(6)
        # column : 2
        X_02, aux_loss[8] = self.up_concat02(X_11, X_00, X_01)#8*(48)
        X_12, aux_loss[9] = self.up_concat12(X_21, T_10, X_11)#16*(24)
        X_22, aux_loss[10] = self.up_concat22(X_31, T_20, X_21)#32*(12)
        # column : 3
        X_03, aux_loss[11] = self.up_concat03(X_12, X_00, X_01, X_02)#8*(48)
        X_13, aux_loss[12] = self.up_concat13(X_22, T_10, X_11, X_12)#16*(24)
        # column : 4
        X_04, aux_loss[13] = self.up_concat04(X_13, X_00, X_01, X_02, X_03)#8*(48)

        # out layer
        out_1 = self.out_1(X_01)#*self.gamma[0]#1*(48)
        out_2 = self.out_2(X_02)#*self.gamma[1]#1*(48)
        out_3 = self.out_3(X_03)#*self.gamma[2]#1*(48)
        out_4 = self.out_4(X_04)#*self.gamma[3]#1*(48)

        out_5 = self.out_5(X_13)#1*(48)
        out_6 = self.out_6(X_22)#1*(48)
        out_7 = self.out_7(X_31)#1*(48)
        out_8 = self.out_8(T_40)#1*(48)

        out = [out_1 , out_2,out_3 ,out_4 ,out_5 ,out_6,out_7,out_8] #1*(48)
        out = torch.stack(out)
        out = out*self.weights
        out=out.sum(0)
        aux_loss=aux_loss.sum(0)

        if self.return_aux_loss:
            return [out, aux_loss]
        else:
            return out
