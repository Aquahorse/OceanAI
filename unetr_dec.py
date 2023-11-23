import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel = 2, stride = 2):
        super().__init__()   
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=0, output_padding=0)
    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class UNETRDecoder(nn.Module):
    def __init__(self, **kwargs): 
        super().__init__()
        self.input_dim = 4 if 'input_dim' not in kwargs else kwargs['input_dim']
        self.output_dim = 3 if 'output_dim' not in kwargs else kwargs['output_dim']
        self.embed_dim = 512 if 'embed_dim' not in kwargs else kwargs['embed_dim']
        self.img_shape = (256, 128, 64) if 'img_shape' not in kwargs else kwargs['img_shape']
        self.patch_size = (16, 16, 8) if 'patch_size' not in kwargs else kwargs['patch_size']
        self.num_heads = 12 if 'num_heads' not in kwargs else kwargs['num_heads']
        self.dropout = 0.1 if 'dropout' not in kwargs else kwargs['dropout']
        self.num_layers = 12 if 'num_layers' not in kwargs else kwargs['num_layers']
        self.ext_layers = [3, 6, 9, 12] if 'ext_layers' not in kwargs else kwargs['ext_layers']

        self.patch_dim = [int(self.img_shape[i] / self.patch_size[i]) for i in range(len(self.img_shape))]

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(self.input_dim, self.embed_dim//16, 3),
                Conv3DBlock(self.embed_dim//16, self.embed_dim//8, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(self.embed_dim, self.embed_dim),
                Deconv3DBlock(self.embed_dim, self.embed_dim//2),
                Deconv3DBlock(self.embed_dim//2, self.embed_dim//4)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(self.embed_dim, self.embed_dim),
                Deconv3DBlock(self.embed_dim, self.embed_dim//2),
            )

        self.decoder9 = \
            Deconv3DBlock(self.embed_dim, self.embed_dim)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(self.embed_dim, self.embed_dim)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(self.embed_dim*2, self.embed_dim),
                Conv3DBlock(self.embed_dim, self.embed_dim),
                Conv3DBlock(self.embed_dim, self.embed_dim),
                SingleDeconv3DBlock(self.embed_dim, self.embed_dim//2)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(self.embed_dim, self.embed_dim//2),
                Conv3DBlock(self.embed_dim//2, self.embed_dim//2),
                SingleDeconv3DBlock(self.embed_dim//2, self.embed_dim//4)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(self.embed_dim//2, self.embed_dim//4),
                Conv3DBlock(self.embed_dim//4, self.embed_dim//4),
                SingleDeconv3DBlock(self.embed_dim//4, self.embed_dim//8, kernel=(2,2,1), stride=(2,2,1)) #DONE: remove the hard coding here
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(self.embed_dim//4, self.embed_dim//8),
                Conv3DBlock(self.embed_dim//8, self.embed_dim//8),
                SingleConv3DBlock(self.embed_dim//8, self.output_dim, 1)
            )
    def reshape_input(self, x):
        x = x.view(-1, self.img_shape[0] // self.patch_size[0], self.img_shape[1] // self.patch_size[1], self.img_shape[2] // self.patch_size[2], self.embed_dim)
        return einops.rearrange(x, 'b h w d c -> b c h w d')
    
    def forward(self, decoder_input:dict):
        z12 = decoder_input['x']
        z0,z3,z6,z9 = decoder_input['z_list']
        z3 = self.reshape_input(z3)
        z6 = self.reshape_input(z6)
        z9 = self.reshape_input(z9)
        z12 = self.reshape_input(z12)      
        # z = z.view(4, -1, self.img_shape[0] // self.patch_size[0], self.img_shape[1] // self.patch_size[1], self.img_shape[2] // self.patch_size[2], self.embed_dim)
        # z = einops.rearrange(z, 'r b h w d c -> r b c h w d')

        # z3, z6, z9, z12 = [z_slice.squeeze(0) for z_slice in z.split(1, dim=0)]

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output
    def self_introduction(self,prefix):
        print(prefix+".UNETR_decoder number of trainable parameters: ", sum(p.numel() for p in self.parameters() if p.requires_grad))
    

if __name__ == '__main__':
    from aiweather_configs import configs
    model = UNETRDecoder(**configs['decoder_configs']['backbone_configs']).to('cuda:3')
    print(model)
    z0 = torch.randn(2,10,128,64,32).to('cuda:3')
    z = torch.randn(3, 2, 128, 512).to('cuda:3')
    z3, z6, z9 = [z_slice.squeeze(0) for z_slice in z.split(1, dim=0)]
    x = torch.randn(2, 8, 4, 2048).to('cuda:3')
    y = model(dict(x=x,z_list = [z0,z3,z6,z9]))
    print(y.shape)