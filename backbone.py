import torch
import math
import torch.nn as nn
def create_2d_positional_embedding(img_height, img_width, d_model):
    # Ensure the model dimension is even
    assert d_model % 2 == 0, "d_model should be even"

    # Create a grid of positions
    y_pos, x_pos = torch.meshgrid(torch.arange(img_height), torch.arange(img_width), indexing='ij')

    # Reshape and scale the position grids
    y_pos, x_pos = y_pos.flatten().float(), x_pos.flatten().float()
    y_pos = (y_pos / (img_height - 1)) * 2 - 1  # Scale to [-1, 1]
    x_pos = (x_pos / (img_width - 1)) * 2 - 1   # Scale to [-1, 1]

    # Initialize the positional encoding matrix
    pos_encoding = torch.zeros((img_height * img_width, d_model)).float()

    # Compute the positional encodings (sin-cos)
    div_term = torch.exp(torch.arange(0, d_model / 2).float() * (-math.log(10000.0) / (d_model / 2)))
    pos_y = torch.ger(y_pos, div_term).view(img_height * img_width, -1)
    pos_x = torch.ger(x_pos, div_term).view(img_height * img_width, -1)

    pos_encoding[:, 0::2] = torch.sin(pos_y)  # Apply sin to even indices
    pos_encoding[:, 1::2] = torch.cos(pos_x)  # Apply cos to odd indices

    # Reshape to the final desired shape
    pos_encoding = pos_encoding.view(img_height, img_width, d_model)

    return pos_encoding

class ImgPosEmbedding(nn.Module):
    def __init__(self,img_height,img_weight,d_model,is_output_layer = False):
        super(ImgPosEmbedding,self).__init__()
        self.img_height = img_height
        self.img_weight = img_weight
        if not is_output_layer:
            init_positional_embedding = create_2d_positional_embedding(img_height, img_weight, d_model).unsqueeze(0).unsqueeze(0) # H*W*d_model -> 1*1*H*W*d_model
        else:
            init_positional_embedding = -create_2d_positional_embedding(img_height, img_weight, d_model).unsqueeze(0) # we want to subtract positional embedding for the output layer
        self.img_positional_encoding = nn.Parameter(init_positional_embedding)
    def forward(self,x):
        return x + self.img_positional_encoding
    
class TimeSeriesPosEmbedding(nn.Module):
    def __init__(self,seq_len,d_model):
        super(TimeSeriesPosEmbedding,self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        init_time_series_embedding = create_2d_positional_embedding(1, seq_len, d_model).unsqueeze(2).unsqueeze(2) # 1*seq_len*d_moel -> 1*seq_len*1*1*d_model
        # print("init_time_series_embedding.shape: ", init_time_series_embedding.shape)
        self.time_series_positional_encoding = nn.Parameter(init_time_series_embedding.flip(1))
    def forward(self,x):
        #assert x.shape == self.time_series_positional_encoding.shape, "x.shape: {}, time_series_positional_encoding.shape: {}".format(x.shape,self.time_series_positional_encoding.shape)
        return x + self.time_series_positional_encoding

class PadUnk(nn.Module):
    def __init__(self,output_height,output_weight,d_model):
        super(PadUnk,self).__init__()
        init_pad_unk = create_2d_positional_embedding(output_height, output_weight, d_model).view(1,output_height*output_weight,d_model) # 1*(o_H*o_W)*d_model
        self.pad_unk = nn.Parameter(init_pad_unk)
    def forward(self,x):
        if len(x.shape) == 5:
            x = x.view(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3],x.shape[4]) # batch_size * (T*H*W) * C
            x = torch.cat((x,self.pad_unk.repeat(x.shape[0],1,1)),dim=1) # batch_size * (T*H*W+o_H*o_W) * C
            return x
        elif len(x.shape) == 3:
            x = torch.cat((x,self.pad_unk.repeat(x.shape[0],1,1)),dim=1)
            return x
            


class TFBackbone(nn.Module):
    def __init__(self,**kwargs):
        '''
            Input: batch_size * T * H * W * C.
                Input is added by a positional encoding (for img) and a positional encoding (for time series).
                Then, the input is flattened to batch_size * (T*H*W) *C, with T*H*W being the sequence length and C being the feature dimension.
            Output: batch_size *(output_height) * (output_width) * C.
        '''
        kwargs['d_model'] = 1440 if 'd_model' not in kwargs else kwargs['d_model']
        kwargs['num_layers'] = 48 if 'num_layers' not in kwargs else kwargs['num_layers']
        kwargs['input_height'] = 6 if 'input_height' not in kwargs else kwargs['input_height']
        kwargs['input_weight'] = 6 if 'input_weight' not in kwargs else kwargs['input_weight']
        kwargs['output_height'] = 6 if 'output_height' not in kwargs else kwargs['output_height']
        kwargs['output_weight'] = 6 if 'output_weight' not in kwargs else kwargs['output_weight']
        kwargs['frame_num'] = 10 if 'frame_num' not in kwargs else kwargs['frame_num']
        kwargs['pad_unk'] = True if 'pad_unk' not in kwargs else kwargs['pad_unk']
        kwargs['num_heads'] = 8
        kwargs['feedforwad_dim'] = 2 * kwargs['d_model']
        kwargs['activation_layer'] = 'leaky_relu'
        
        super(TFBackbone,self).__init__()
        
        
        self.d_model = kwargs['d_model']
        self.num_layers = kwargs['num_layers']
        self.input_height = kwargs['input_height']
        self.input_weight = kwargs['input_weight']
        self.output_height = kwargs['output_height']
        self.output_weight = kwargs['output_weight']
        self.frame_num = kwargs['frame_num']
        self.pad_unk = kwargs['pad_unk']
        
        if kwargs['activation_layer'] == 'leaky_relu':
            activation_layer = torch.nn.functional.leaky_relu
        elif kwargs['activation_layer'] == 'relu':
            activation_layer = torch.nn.functional.relu
        if not self.pad_unk:
            assert self.output_height == self.input_height and self.output_weight == self.input_weight
        
        self.img_positional_encoding = ImgPosEmbedding(self.input_height,self.input_weight, self.d_model)
        self.output_img_positional_encoding = ImgPosEmbedding(self.output_height, self.output_weight, self.d_model,is_output_layer = True)
        self.time_series_positional_encoding = TimeSeriesPosEmbedding(self.frame_num,self.d_model)
        self.pad_unk_layer = PadUnk(self.output_height,self.output_weight,self.d_model) if self.pad_unk else None
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.d_model, kwargs['num_heads'], kwargs['feedforwad_dim'], batch_first=True, activation=activation_layer), self.num_layers)
        
    def forward(self,encoded_img):
        assert encoded_img.shape[1] == self.frame_num and encoded_img.shape[2] == self.input_height and encoded_img.shape[3] == self.input_weight
        batch_size = encoded_img.shape[0]
        #print("encoded_img.shape: ", encoded_img.shape)
        encoded_img_with_embeddings = self.img_positional_encoding(self.time_series_positional_encoding(encoded_img)) # these two can be reversed
        input_embeddings = encoded_img_with_embeddings.view(batch_size,encoded_img_with_embeddings.shape[1]*encoded_img_with_embeddings.shape[2]*encoded_img_with_embeddings.shape[3],encoded_img_with_embeddings.shape[4]) # batch_size * (T*H*W) * C
        if self.pad_unk:
            input_embeddings = self.pad_unk_layer(input_embeddings)
        output_embeddings = self.transformer_encoder(input_embeddings)
        output_embeddings = output_embeddings[:, -self.output_height*self.output_weight:, :].view(batch_size,self.output_height,self.output_weight,self.d_model)
        return self.output_img_positional_encoding(output_embeddings) # we subtract the positional encoding here because we think we add it previously.
    def param_counting(self):
        return dict(
            Img_positional_encoding_learnable_parameters = sum(p.numel() for p in self.img_positional_encoding.parameters() if p.requires_grad),
            Time_series_positional_encoding_learnable_parameters = sum(p.numel() for p in self.time_series_positional_encoding.parameters() if p.requires_grad),
            Pad_unk_learnable_parameters = sum(p.numel() for p in self.pad_unk_layer.parameters() if p.requires_grad) if self.pad_unk else 0,
            TF_layer_learnable_parameters = sum(p.numel() for p in self.transformer_encoder.parameters() if p.requires_grad),
            TF_layer_total_parameters = sum(p.numel() for p in self.transformer_encoder.parameters()),
            Output_Img_positional_encoding_learnable_parameters = sum(p.numel() for p in self.output_img_positional_encoding.parameters() if p.requires_grad),
            Total_learnable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad),
            Total_parameters = sum(p.numel() for p in self.parameters())
        )
    def self_introduction(self,prefix=''):
        for key,value in self.param_counting().items():
            print(prefix+'.TFBackbone.', key, ": ", value)
        
        
if __name__ == '__main__':
    from aiweather_configs import configs
    tfbackbone = TFBackbone(**configs['backbone_configs'])
    tfbackbone = tfbackbone.to('cuda:3')
    # print(tfbackbone)
    x = torch.randn(2,10,8,4,2048).to('cuda:3')
    while True:
        tfbackbone.zero_grad()
        y = tfbackbone(x)
        print(y.shape)
        loss = y.sum()
        loss.backward()
    # print(create_2d_positional_embedding(6,6,4))
    
