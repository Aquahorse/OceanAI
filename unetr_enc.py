import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PatchEmbedding(nn.Module):
    def __init__(self,input_dim,embed_dim,cube_size,patch_size):
        super().__init__()
        self.direct_embed = nn.Conv3d(input_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        for item in patch_size:
            assert math.log2(item).is_integer(), "Patch size should be a power of 2."
        def end(patch_list):
            return 1 in patch_list
        patch_size_list = copy.deepcopy(patch_size)
        self.indirect_embed = nn.ModuleList()
        init = True
        while not end(patch_size_list):
            self.indirect_embed.append(nn.Conv3d(input_dim if init else embed_dim, embed_dim, kernel_size=(3,3,3), stride=(2,2,2),padding=(1,1,1)))
            init = False
            patch_size_list = [i//2 for i in patch_size_list]
            if 1 not in patch_size_list:
                self.indirect_embed.append(nn.LeakyReLU())
        self.indirect_embed.append(nn.Conv3d(embed_dim, embed_dim, kernel_size=patch_size_list, stride=patch_size_list))
    def forward(self,x):
        direct_embed = self.direct_embed(x)
        indirect_embed = x
        # print(direct_embed.shape)
        # print("AAAAAAAAAAAAAAA")
        for layer in self.indirect_embed:
            # print(indirect_embed.shape)
            indirect_embed = layer(indirect_embed)
        # assert False
        return direct_embed + indirect_embed
        
class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif isinstance(patch_size, (tuple, list)):
            assert len(patch_size) == 3, "Patch size should be a tuple or list of 3 integers."
        if isinstance(cube_size, int):
            cube_size = (cube_size, cube_size, cube_size)
        elif isinstance(cube_size, (tuple, list)):
            assert len(cube_size) == 3, "Cube size should be a tuple or list of 3 integers."
            
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size[0] * patch_size[1] * patch_size[2]))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.patch_embeddings = PatchEmbedding(input_dim, embed_dim, cube_size, patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim)) #TODO: convert to 3d positional embedding
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        #x: (B, C, H, W, D) 
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size[0] * patch_size[1] * patch_size[2]))
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, cube_size, patch_size)
            self.layer.append(copy.deepcopy(layer))
        
        print("number of trainable parameter in embedding layer is ", sum(p.numel() for p in self.embeddings.parameters() if p.requires_grad))
        print("number of trainable parameter in transformer layer is ", sum(p.numel() for p in self.layer.parameters() if p.requires_grad))
        
    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers


class UNETR(nn.Module):
    #def __init__(self, img_shape=(128, 128, 128), input_dim=4, output_dim=3, embed_dim=768, patch_size=16, num_heads=12, dropout=0.1):
    def __init__(self, **kwargs): 
        super().__init__()
        self.input_dim = 4 if 'input_dim' not in kwargs else kwargs['input_dim']
        self.output_dim = 3 if 'output_dim' not in kwargs else kwargs['output_dim']
        self.embed_dim = 256 if 'embed_dim' not in kwargs else kwargs['embed_dim']
        self.img_shape = (256, 128, 64) if 'img_shape' not in kwargs else kwargs['img_shape']
        self.patch_size = (16, 16, 8) if 'patch_size' not in kwargs else kwargs['patch_size']
        self.num_heads = 12 if 'num_heads' not in kwargs else kwargs['num_heads']
        self.dropout = 0.1 if 'dropout' not in kwargs else kwargs['dropout']
        self.num_layers = 12 if 'num_layers' not in kwargs else kwargs['num_layers']
        self.ext_layers = [3, 6, 9, 12] if 'ext_layers' not in kwargs else kwargs['ext_layers']

        self.patch_dim = [int(self.img_shape[i] / self.patch_size[i]) for i in range(len(self.img_shape))]

        # Transformer Encoder
        self.transformer = \
            Transformer(
                self.input_dim,
                self.embed_dim,
                self.img_shape,
                self.patch_size,
                self.num_heads,
                self.num_layers,
                self.dropout,
                self.ext_layers
            )

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(self.input_dim, 32, 3),
                Conv3DBlock(32, 64, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(self.embed_dim, 512),
                Deconv3DBlock(512, 256),
                Deconv3DBlock(256, 128)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(self.embed_dim, 512),
                Deconv3DBlock(512, 256),
            )

        self.decoder9 = \
            Deconv3DBlock(self.embed_dim, 512)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(self.embed_dim, 512)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(1024, 512),
                Conv3DBlock(512, 512),
                Conv3DBlock(512, 512),
                SingleDeconv3DBlock(512, 256)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(512, 256),
                Conv3DBlock(256, 256),
                SingleDeconv3DBlock(256, 128)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(256, 128),
                Conv3DBlock(128, 128),
                SingleDeconv3DBlock(128, 64, kernel=(2,2,1), stride=(2,2,1)) #TODO: remove the hard coding here
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                SingleConv3DBlock(64, self.output_dim, 1)
            )

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

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
    
class UnetREncoder(nn.Module):
    def __init__(self, **kwargs): 
        '''
            Input:
                (B*T) * C * H * W * D
            Output:
                (B*T) * (H/Ph * W/Pw) * (D/Pd * C)
        '''
        super().__init__()
        self.input_dim = 4 if 'input_dim' not in kwargs else kwargs['input_dim']
        self.output_dim = 3 if 'output_dim' not in kwargs else kwargs['output_dim']
        self.embed_dim = 256 if 'embed_dim' not in kwargs else kwargs['embed_dim']
        self.img_shape = (256, 128, 64) if 'img_shape' not in kwargs else kwargs['img_shape']
        self.patch_size = (16, 16, 8) if 'patch_size' not in kwargs else kwargs['patch_size']
        self.num_heads = 12 if 'num_heads' not in kwargs else kwargs['num_heads']
        self.dropout = 0.1 if 'dropout' not in kwargs else kwargs['dropout']
        self.num_layers = 12 if 'num_layers' not in kwargs else kwargs['num_layers']
        self.ext_layers = [3, 6, 9, 12] if 'ext_layers' not in kwargs else kwargs['ext_layers']

        self.patch_dim = [int(self.img_shape[i] / self.patch_size[i]) for i in range(len(self.img_shape))]

        # Transformer Encoder
        self.transformer = \
            Transformer(
                self.input_dim,
                self.embed_dim,
                self.img_shape,
                self.patch_size,
                self.num_heads,
                self.num_layers,
                self.dropout,
                self.ext_layers
            )

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z #Prepare for skip connections
        z12 = z12.view(-1, (self.img_shape[0]) // (self.patch_size[0]), (self.img_shape[1]) // (self.patch_size[1]), (self.embed_dim * self.img_shape[2])// self.patch_size[2])
        return z0, z3, z6, z9, z12 
    def self_introduction(self,prefix):
        print(prefix+".UNETR_encoder "+"number of trainable parameter is ", sum(p.numel() for p in self.transformer.parameters() if p.requires_grad))

if __name__ == '__main__':
    from aiweather_configs import configs
    model = UnetREncoder(**configs['encoder_configs']['backbone_configs']).to('cuda:3')
    # print(model)
    x = torch.randn(20,10,128,64,32).to('cuda:3')
    print("number of trainable model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    while True:
        model.zero_grad()
        print(x.shape)
        y = model(x)
        for i in y:
            print(i.shape)
        # print(y.shape)
        # y.sum().backward()
        break