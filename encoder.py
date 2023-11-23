import torch
import torch.nn as nn
import einops
from unetr_enc import UnetREncoder
import math
class Encoder(nn.Module):
    def __init__(self,**kwargs):
        '''
        Input: tensor batch_size * T * C * H * W * D.
        Output: dict:
                    x: batch_size * T * H_o * W_o * D_o
                    z_list: list of z0,z3,z6,z9.
                        z0: batch_size * H_o * W_o * D_o
                        z3 through z9: batch_size * (H_o*W_o) * D_o
        '''
        super().__init__()
        assert 'backbone_configs' in kwargs
        kwargs['norm_layer'] = 'LayerNorm' if 'norm_layer' not in kwargs else kwargs['norm_layer']
        self.interpolate_to_shape = kwargs['interpolate_layer']['output_size']
        if kwargs['norm_layer'] == 'LayerNorm':
            self.norm_layer = nn.LayerNorm(kwargs['interpolate_layer']['output_size']) # do not norm over the channel layer.
        else:
            assert False
        
        self.encoder_backbone = UnetREncoder(**kwargs['backbone_configs'])
    def obtain_final(self,zi,b):
        zi = einops.rearrange(zi, '(b t) ... -> t b ...', b=b)
        zi = zi[-1]
        return zi
    def forward(self,x):
        # assert x.shape[-2] == 2*x.shape[-1]
        b = x.shape[0]
        x = einops.rearrange(x, 'b t c h w d -> (b t) c h w d')
        x = torch.nn.functional.interpolate(x, size=self.interpolate_to_shape, mode='trilinear')
        x = self.norm_layer(x)
        # x = torch.cat(x[..., :x.shape[2]//2, :], x[..., x.shape[2]//2:, :], dim=-3)
        
        
        # x = einops.rearrange(x, 'x c h w -> x h w c')
        z0,z3,z6,z9,z12 = self.encoder_backbone(x)
        z0 = self.obtain_final(z0,b)
        z3 = self.obtain_final(z3,b)
        z6 = self.obtain_final(z6,b)
        z9 = self.obtain_final(z9,b)
        
        bt, h, w, new_c = z12.shape
        x = einops.rearrange(z12, '(b t) h w c -> b t h w c', b=b)
        return dict(
                    x=x,
                    z_list=[z0,z3,z6,z9]
                )
    def self_introduction(self,prefix):
        return self.encoder_backbone.self_introduction(prefix+'.encoder_backbone')

if __name__ == '__main__':
    from aiweather_configs import configs
    encoder = Encoder(**configs['encoder_configs']).to('cuda:3')
    # print(encoder)
    x = torch.randn(2,10, 10, 256,128,32).to('cuda:3')
    print("number of trainable encoder parameters: ", sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    while True:
        encoder.zero_grad()
        y = encoder(x)
        for item in y:
            print(item.shape)
        break