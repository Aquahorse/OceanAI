import torch
import torch.nn as nn
import math
from encoder import Encoder
from decoder import Decoder
from backbone import TFBackbone

class TransFormerWeather(nn.Module):
    def __init__(self,**kwargs):
        '''
            Input: input dict: x = tensor batch_size * T * C * H * W * D.
            Output: tensor batch_size * C * H * W * D.
        '''
        super().__init__()
        self.encoder = Encoder(**kwargs['encoder_configs'])
        self.backbone = TFBackbone(**kwargs['backbone_configs'])
        self.decoder = Decoder(**kwargs['decoder_configs'])
        self.self_introduction()
    def forward(self,input_dict):
        x = input_dict['x']
        land_mask = input_dict['land_mask']
        if land_mask is not None:
            x = torch.cat([x,land_mask],dim=2)
        encoded_dict = self.encoder(x)
        x = encoded_dict['x']
        x = self.backbone(x)
        decoded_x = self.decoder(dict(
            x=x,
            z_list = encoded_dict['z_list']
        ))
        return decoded_x
    def self_introduction(self,prefix=None):
        if prefix is None:
            prefix = self.__class__.__name__
            print('--------------------------------------------------------------Encoder Description Begin--------------------------------------------------------------')
        self.encoder.self_introduction(prefix+'.encoder')
        print('--------------------------------------------------------------Backbone Description Begin--------------------------------------------------------------')
        self.backbone.self_introduction(prefix+'.backbone')
        print('--------------------------------------------------------------Decoder Description Begin--------------------------------------------------------------')
        self.decoder.self_introduction(prefix+'.decoder')
        print('--------------------------------------------------------------Over all:--------------------------------------------------------------')
        print(prefix+' total number of trainable parameters: ', sum(p.numel() for p in self.parameters() if p.requires_grad))
        print(prefix+' total number of parameters: ', sum(p.numel() for p in self.parameters()))

if __name__ == '__main__':
    from aiweather_configs_sjj import get_configs
    configs = get_configs('tiny')
    model = TransFormerWeather(**configs).to('cuda:3')
    # print(model)
    x = torch.randn(8,10, 10, 256,128,32).to('cuda:3')
    land_mask = torch.randn(8,10,1,256,128,32).to('cuda:3')
    print("number of trainable model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    while True:
        model.zero_grad()
        y = model(dict(x=x,land_mask=land_mask))
        print(y.shape)
        y.sum().backward()
        break