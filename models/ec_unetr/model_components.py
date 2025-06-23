from base64 import encode
from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from models.ec_unetr.layers import LayerNorm
from models.ec_unetr.transformerblock import TransformerBlock
from models.ec_unetr.dynunet_block import get_conv_layer, UnetResBlock


einops, _ = optional_import("einops")

class StageEncoder(nn.Module):
    def __init__(self, in_channels,out_channels,input_size,
                 proj_size , depth,  num_heads, spatial_dims=3, 
                 dropout=0.0, transformer_dropout_rate=0.1 ,conv_encoder=False):
        super().__init__()

        #downsample: conv+norm
        self.downsample_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=out_channels),
        )

        #stage blocks
        stage_blocks = []
        if conv_encoder == True:
            self.stage_blocks.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=3, stride=1,
                             norm_name='instance', ))
        else:
            #include depth times TransformerBlock       
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=input_size, hidden_size=out_channels,
                                                        proj_size=proj_size, num_heads=num_heads,
                                                        dropout_rate=transformer_dropout_rate, pos_embed=True))  
        self.stage_blocks=nn.Sequential(*stage_blocks)
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
        x = self.stage_blocks(x)
        return x

class StageDecoder(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

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

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels,
                                                     proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)
        return out

class UnetrEncoder(nn.Module):
    def __init__(self, input_size,dims, proj_size, depths,  num_heads=4):
        super().__init__()

        self.encode_blocks=nn.ModuleList()
        self.stage_num=len(input_size)
        for i in range(self.stage_num):
            encoder = StageEncoder(dims[i],dims[i+1],input_size[i],proj_size[i],depths[i],num_heads)
            self.encode_blocks.append(encoder)

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
        hidden_states = []
        for i in range(self.stage_num):
            x=self.encode_blocks[i](x)
            hidden_states.append(x)
        return hidden_states

class UnetrDecoder(nn.Module):
    def __init__(self, input_size,dims, proj_size, depths,  num_heads=4):
        super().__init__()

        input_size=input_size[::-1]
        dims=dims[::-1]

        self.decode_blocks=nn.ModuleList()
        self.stage_num=len(input_size)-1
        for i in range(self.stage_num):
            decoder = StageDecoder(
                spatial_dims=3,
                in_channels=dims[i],
                out_channels=dims[i+1],
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name='instance',
                out_size=input_size[i+1]
            )
            self.decode_blocks.append(decoder)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, enc_states):
        dec_states=[]
        dec_states.append(enc_states[-1])
        for i in range(self.stage_num):
            dec=self.decode_blocks[i](dec_states[i], enc_states[-i-2])
            dec_states.append(dec)
        return dec_states

