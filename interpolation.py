import h5py
import os

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Temperature for the two consecutive timestamp
temperature_fields = []
timestamps = []

# Folder Path
path = "/scratch/kr97/xh7958/comp4560/solutions"
  
# Read text File  
def read_text_file(file_path):
    with h5py.File(file_path, 'r') as f:
        temperature_fields.append(f['temperature'][:])
        timestamps.append(f['timestamps'][:])
        
# Iterate through all file
file_count = 0
for file in os.listdir(path):
    file_path = f"{path}/{file}"
  
    # call read text file function
    read_text_file(file_path)
    #print(f"{file_path} is finished reading")
    file_count = file_count + 1
    

temperature_fields = np.asarray(temperature_fields)
timestamps = np.asarray(timestamps)


# single file Interpolation (the first file for example)
from scipy.interpolate import LinearNDInterpolator

new_temperature_fields = []

x = np.tile(np.repeat(list(range(201)),401),100)
y = np.tile(np.tile(list(range(401)),201),100)
standard_timestamps = np.repeat(np.linspace(np.min(timestamps[:,0]), np.max(timestamps[:,99]), 100),201*401)

for i in range(file_count):
    true_timestamp = np.repeat(timestamps[i],201*401)
    interpolating_function = LinearNDInterpolator((true_timestamp,x,y), temperature_fields[i].flatten())
    
    points = np.array(np.dstack((standard_timestamps,x,y)))
    standard_temperature_field = interpolating_function(points)
    
    new_temperature_fields.append(standard_temperature_field.reshape(100,201,401))
    
new_temperature_fields = np.asarray(new_temperature_fields)

with open('/scratch/kr97/xh7958/comp4560/solutions_standard.npy', 'wb') as f:
    np.save(f, new_temperature_fields)


