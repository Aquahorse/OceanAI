import copy

def update_dict(d : dict, t: dict):
    d  = copy.deepcopy(d)
    for k,v in t.items():
        d[k] = v
    return d


base_configs = dict(
    data_configs = dict(
        data_dir = "/home/LVM_date2/data/OceanAI",
        sequence_length = 10, #TODO: modify it
        batch_size = 8,
        split_ratio = [0.8, 0.1, 0.1],
    ),
)
large_unetr_configs = dict(
    input_dim = 11,
    output_dim = 10,
    embed_dim = 512,
    img_shape = (128, 64, 32),
    patch_size = (16, 16, 8),
    num_heads = 8,
    dropout = 0.1,
    num_layers = 12,
    ext_layers = [3,6,9,12],
)

large_model_configs = dict(
    encoder_configs = dict(

        interpolate_layer = dict(
            name = 'bilinear',
            input_size = [256,128,32],
            output_size = [128,64,32],
        ),
        
        norm_layer = 'LayerNorm',
        
        backbone_configs = update_dict(large_unetr_configs, {'name':'unetr_encoder_only'})
    ),
    backbone_configs = dict(
        d_model = 2048,
        num_layers = 48,
        input_height = 8,
        input_weight = 4,
        output_height = 8,
        output_weight = 4,
        frame_num = 10,
        pad_unk = True,
        num_heads = 8,
        feedforwad_dim = 4096,
        activation_layer = 'leaky_relu'
    ),
    decoder_configs = dict(
        backbone_configs = update_dict(large_unetr_configs, {'name':'unetr_decoder_only'})
    ),
)

base_unetr_configs = dict(
    input_dim = 11,
    output_dim = 10,
    embed_dim = 256,
    img_shape = (128, 64, 32),
    patch_size = (16, 16, 8),
    num_heads = 8,
    dropout = 0.1,
    num_layers = 12,
    ext_layers = [3,6,9,12],
)

base_model_configs = dict(
    encoder_configs = dict(

        interpolate_layer = dict(
            name = 'bilinear',
            input_size = [256,128,32],
            output_size = [128,64,32],
        ),
        
        norm_layer = 'LayerNorm',
        
        backbone_configs = update_dict(base_unetr_configs, {'name':'unetr_encoder_only'})
    ),
    backbone_configs = dict(
        d_model = 1024,
        num_layers = 36,
        input_height = 8,
        input_weight = 4,
        output_height = 8,
        output_weight = 4,
        frame_num = 10,
        pad_unk = True,
        num_heads = 8,
        feedforwad_dim = 2048,
        activation_layer = 'leaky_relu'
    ),
    decoder_configs = dict(
        backbone_configs = update_dict(base_unetr_configs, {'name':'unetr_decoder_only'})
    ),
)

small_unetr_configs = dict(
    input_dim = 11,
    output_dim = 10,
    embed_dim = 128,
    img_shape = (128, 64, 32),
    patch_size = (16, 16, 8),
    num_heads = 8,
    dropout = 0.1,
    num_layers = 12,
    ext_layers = [3,6,9,12],
)

small_model_configs = dict(
    encoder_configs = dict(

        interpolate_layer = dict(
            name = 'bilinear',
            input_size = [256,128,32],
            output_size = [128,64,32],
        ),
        
        norm_layer = 'LayerNorm',
        
        backbone_configs = update_dict(small_unetr_configs, {'name':'unetr_encoder_only'})
    ),
    backbone_configs = dict(
        d_model = 512,
        num_layers = 24,
        input_height = 8,
        input_weight = 4,
        output_height = 8,
        output_weight = 4,
        frame_num = 10,
        pad_unk = True,
        num_heads = 8,
        feedforwad_dim = 1024,
        activation_layer = 'leaky_relu'
    ),
    decoder_configs = dict(
        backbone_configs = update_dict(small_unetr_configs, {'name':'unetr_decoder_only'})
    ),
)

tiny_unetr_configs = dict(
    input_dim = 11,
    output_dim = 10,
    embed_dim = 64,
    img_shape = (128, 64, 32),
    patch_size = (16, 16, 8),
    num_heads = 8,
    dropout = 0.1,
    num_layers = 12,
    ext_layers = [3,6,9,12],
)

tiny_model_configs = dict(
    encoder_configs = dict(

        interpolate_layer = dict(
            name = 'bilinear',
            input_size = [256,128,32],
            output_size = [128,64,32],
        ),
        
        norm_layer = 'LayerNorm',
        
        backbone_configs = update_dict(tiny_unetr_configs, {'name':'unetr_encoder_only'})
    ),
    backbone_configs = dict(
        d_model = 256,
        num_layers = 18,
        input_height = 8,
        input_weight = 4,
        output_height = 8,
        output_weight = 4,
        frame_num = 10,
        pad_unk = True,
        num_heads = 4,
        feedforwad_dim = 512,
        activation_layer = 'leaky_relu'
    ),
    decoder_configs = dict(
        backbone_configs = update_dict(tiny_unetr_configs, {'name':'unetr_decoder_only'})
    ),
)

model_config_dict = dict(
    tiny = tiny_model_configs,
    small = small_model_configs,
    base = base_model_configs,
    large = large_model_configs,
    
    mini = tiny_model_configs,
    medium = base_model_configs,
    
)


wandb_configs = dict(
    wandb_entity = 'shijingzhe',
    wandb_project = 'oceanbigmodel'
)





training_configs = dict(
    seed = 1145,
    checkpoint_dir = '/share_ssd/AI_Plus_X_weather/oceanbigmodel/checkpoints',
    load_from = None, 
    lr = 1e-4,
    weight_decay = 1e-5,
    max_epochs = 200,
    gpus = [0,1,2,3,4,5,6,7,],
)



def get_configs(model_size='base'):
    # large: 20GB memory for batchsize = 1
    # base: 21GB memory for batchsize = 5
    # small: 21GB memory for batchsize = 8
    
    # model_size v.s. number of parameters in Encoder, Backbone, Decoder
    # large: 65M, 1.61B, 72M
    # base: 25M, 303M, 18M
    # small: 11M, 50M, 5M
    # tiny: 5M, 9M, 1M
    assert model_size in ['tiny','small','base','large',  'mini','medium'] # mini == tiny, medium == base
    model_configs = model_config_dict[model_size]
    

    wandb_configs['wandb_project'] = wandb_configs['wandb_project'] + '_____model_size_'+model_size
    _configs = update_dict(base_configs, model_configs)
    train_configs = update_dict(training_configs, wandb_configs)
    _configs = update_dict(_configs, train_configs)
    return _configs

configs = get_configs()



