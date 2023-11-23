import pytorch_lightning as pl
import torch
from torch import nn

from encoder import Encoder
from backbone import TFBackbone
from oceanbigmodel.unetr_dec import UNETRDecoder
from model import TransFormerWeather

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import time
import os
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

import wandb
from dataloader import OceanDataModule
import datetime

class Plmodel(pl.LightningModule):
    def __init__(self, model,**kwargs):
        super().__init__()
        self.model = model
        self.lr = kwargs['lr']
        self.weight_decay = kwargs['weight_decay']

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Implement your training logic here
        input_sequence = batch['input_sequence']
        land_mask = batch['land_mask']
        target_sequence = batch['target_sequence']
        target_land_mask = batch['target_land_mask']
        
        outputs = self.model(dict(
            x = input_sequence,
            land_mask = land_mask,
        ))
        loss = self.loss_function(outputs, targets, target_land_mask)
        return loss

    def validation_step(self, batch, batch_idx):
        # Implement your training logic here
        input_sequence = batch['input_sequence']
        land_mask = batch['land_mask']
        target_sequence = batch['target_sequence']
        
        outputs = self.model(dict(
            x = input_sequence,
            land_mask = land_mask,
        ))
        loss = self.loss_function(outputs, target_sequence)
        return loss
        # Implement validation logic here if needed
        pass
    def test_step(self,batch,batch_idx):
        # Implement your training logic here
        input_sequence = batch['input_sequence']
        land_mask = batch['land_mask']
        target_sequence = batch['target_sequence']
        
        outputs = self.model(dict(
            x = input_sequence,
            land_mask = land_mask,
        ))
        loss = self.loss_function(outputs, targets)
        return loss
    def configure_optimizers(self):
        # Configure your optimizers and learning rate schedulers here
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        return optimizer

    def loss_function(self, outputs, targets,target_land_mask):
        # Define your loss function here
        return nn.functional.mse_loss(outputs*target_land_mask, targets*target_land_mask)
    
class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device)
        torch.cuda.synchronize(trainer.strategy.root_device)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_modul):
        torch.cuda.synchronize(trainer.strategy.root_device)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass
        
class SetupCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.ckptdir= kwargs['checkpoint_dir']
        self.logdir = kwargs['log_dir']
        self.resume = kwargs['resume_training'] if 'resume_training' in kwargs else False
        self.now = time.strftime("%Y-%m-%d-%H-%M-%S")

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            trainer.save_checkpoint(os.path.join(self.ckptdir, 'checkpoints'))

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
        else:
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass
    def on_train_start(self, trainer, pl_module):
        print("Training Arguments:")
        for arg, value in vars(self.kwargs).items():
            print(f"{arg}:\t {value}")

    
def main(args):
    pl.seed_everything(args['seed'] if 'seed' in args else 1145)
    print("Running experiments with args:")
    print(args)
    # Init our model
    model = TransFormerWeather(**args)
    plmodel = Plmodel(model,**args)
    if args['load_from'] is not None:
        plmodel = Plmodel.load_from_checkpoint(args['load_from'], model=model, params = args)
    
    data_module = OceanDataModule(**args)
    
    now = datetime.datetime.now()
    timestr=now.strftime("%Y%m%d-%H%M%S")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'min',
        dirpath = args['checkpoint_dir'],
        save_top_k = 3,
        every_n_epochs = 1,
    )
    
    cuda_callback = CUDACallback()
    setup_callback = SetupCallback(**args)
    
    wandb_logger = WandbLogger(project=args['wandb_project'],
                               name='runtime:'+timestr,
                               entity=args['wandb_entity'])
    wandb_logger.watch(plmodel,log='all')
    from pytorch_lightning.strategies import DDPStrategy

    ddp = DDPStrategy(process_group_backend="nccl")
    
    trainer=pl.Trainer(callbacks=[checkpoint_callback],logger=wandb_logger,
                       accelerator='gpu',
                       devices=args['gpus'],
                       strategy=ddp,
                       max_epochs=args['max_epochs'],precision=32)
    
    trainer.fit(model,data_module)


if __name__ == '__main__':
    from aiweather_configs import get_configs
    configs = get_configs(model_size='large')
    main(configs)