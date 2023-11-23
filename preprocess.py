import os
import re
import netCDF4
import numpy as np
import torch

def load_and_concatenate_data(folder_path):
    data_dict = {}
    files = sorted(os.listdir(folder_path))

    for file in files:
        if file.endswith('.nc'):
            match = re.match(r"(.+)_\w+_\w+_\w+_\w+_(\d{6})-\d{6}\.nc", file)
            if match:
                variable_name, year = match.groups()

                # Load the data
                file_path = os.path.join(folder_path, file)
                with netCDF4.Dataset(file_path, mode='r') as nc_file:
                    data = nc_file[variable_name][:]

                    # Check data shape
                    if data.shape[0] == 12:
                        if variable_name not in data_dict:
                            data_dict[variable_name] = []
                        data_dict[variable_name].append((year, data))

    # Concatenating data sorted by year along dimension 0
    for variable in data_dict:
        sorted_data = sorted(data_dict[variable], key=lambda x: x[0])
        concatenated_data = np.concatenate([data for _, data in sorted_data], axis=0)
        data_dict[variable] = concatenated_data

    return data_dict


if __name__ == '__main__':
    folder_path = '/home/LVM_date2/data/OceanAI'
    dir_names = os.listdir(folder_path)
    for dir_name in dir_names:
        data_path = os.path.join(folder_path, dir_name)
        data_dict = load_and_concatenate_data(data_path)
        print(f"The variables for {dir_name} are {data_dict.keys()}")
        torch.save(f'/home/LVM_date2/maqw/data/{dir_name}_data.pth', data_dict)
    
