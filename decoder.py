import torch
import torch.nn as nn
import einops
from unetr_dec import UNETRDecoder
import math
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        '''
            Input: dict:
                    x: batch_size * Hb * Wb * token_dim
                    z_list: list of z0,z3,z6,z9.
                        z0: batch_size * H_o * W_o * D_o
                        z3 through z9: batch_size * (H_o*W_o) * D_o
            Output: tensor batch_size * Hi * Wi *D
        '''
        super().__init__()
        assert 'backbone_configs' in kwargs
        self.decoder_backbone = UNETRDecoder(**kwargs['backbone_configs'])
    def forward(self,input_dict):
        x = self.decoder_backbone(input_dict)
        return x
    def self_introduction(self,prefix):
        return self.decoder_backbone.self_introduction(prefix+'.decoder_backbone')