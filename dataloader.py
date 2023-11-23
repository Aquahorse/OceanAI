import torch
import os
# Define a dataloader
from torch.utils.data import DataLoader
import netCDF4 as nc

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split

# nc_path = 'data_sample.nc'

index_oras5 = ['vozocrtx', 'vomecrty', 'votemper', 'vosaline', 'sohefldo', 
               'sosaline', 'sosstsst', 'sossheig', 'sozotaux', 'sometauy']

index_cmip6 = ['uo', 'vo', 'thetao', 'so', 'hfds',
               'sos', 'tos', 'zos', 'tauuo', 'tauvo']


# # Open the netCDF file
# ds = nc.Dataset(nc_path, 'r')  

def get_paths_from_folder(folder_path):
    file_list = os.listdir(folder_path)
    file_list.sort()
    file_list = [os.path.join(folder_path,file) for file in file_list]
    return file_list

class OceanDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        assert 'data_dir' in kwargs, "data directory is not provided."
        self.nc_file_list = get_paths_from_folder(kwargs['data_dir'])
        self.sequence_length = kwargs['sequence_length'] if 'sequence_length' in kwargs else 10
        # Create a dataset containing all the data, each item should correspond to a single month (each file corresponds to a year and dimension 0 is time)
        self.dataset = nc.MFDataset(self.nc_file_list, aggdim='time')
        safe_len = len(self.data) - self.sequence_length - 3
        self.random_shuffle_list = torch.randperm(len(safe_len))
        # self.random_shuffle_list.shape = (len(safe_len),)
    
    def __len__(self):
        return len(self.data) - self.sequence_length - 3 # something > 1?2? 3 would be safe anyway

    def __getitem__(self, before_idx):
        idx = self.random_shuffle_list[before_idx]
        # Generate sequences
        input_sequence = self.data[idx:idx+self.sequence_length, :, :, :]
        #target_sequence = self.data[idx+1:idx+self.sequence_length+1, :, :, :]
        target_sequence = self.data[idx+self.sequence_length+1:idx+self.sequence_length+2, :, :, :].unsqueeze(0)
        return input_sequence, target_sequence

def collate_fn(batch_list):
    input_sequence_batch = torch.stack([item[0] for item in batch_list],dim=0)
    target_sequence_batch = torch.stack([item[1] for item in batch_list],dim=0)
    land_mask = torch.zeros(input_sequence_batch.shape[0]+[1]+input_sequence_batch.shape[2:], dtype=torch.float32,device=input_sequence_batch.device)
    land_mask[batch>=1e20-10] = 1.0
    batch[batch>=1e20-10] = 0.0
    return dict(
        input_sequence = input_sequence_batch,
        land_mask = land_mask,
        target_sequence = target_sequence_batch
    )
    
class OceanDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        dataset = OceanDataset(**kwargs)
        self.num_workers = kwargs['num_workers'] if 'num_workers' in kwargs else 10
        self.pin_memory = kwargs['pin_memory'] if 'pin_memory' in kwargs else True
        self.drop_last = kwargs['drop_last'] if 'drop_last' in kwargs else True
        split_ratio = kwargs['split_ratio'] if 'split_ratio' in kwargs else [0.8, 0.1, 0.1]
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 8
        self.trainset, self.valset, self.testset = random_split(dataset, [int(len(dataset)*ratio) for ratio in split_ratio])
        
        
    def setup(self, stage=None):
        return 

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last)
    
