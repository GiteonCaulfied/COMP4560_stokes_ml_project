import h5py
import os

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torch.utils.data.sampler import SubsetRandomSampler


# Temperature for the two consecutive timestamp
temperature_fields = []

# Folder Path
path = "/scratch/kr97/xh7958/comp4560/solutions"
encoder_path = "Conv2D_encoder_best_Gadi.pth"
decoder_path = "Conv2D_decoder_best_Gadi.pth"
  
# Read text File  
def read_text_file(file_path):
    with h5py.File(file_path, 'r') as f:
        temperature_fields.append(f['temperature'][:])
        
        
# Iterate through all file
file_count = 0
for file in os.listdir(path):
    file_path = f"{path}/{file}"
  
    # call read text file function
    read_text_file(file_path)
    #print(f"{file_path} is finished reading")
    file_count = file_count + 1

timestamps_number = temperature_fields[0].shape[0]
temperature_fields = np.asarray(temperature_fields).reshape(file_count*timestamps_number,201,401)


#Parameters
n_epoch = 1000
batch_size = 16
lr = 5e-5
accurate_loss_baseline = 0.005

# Check current device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
    
print("Current device is ",device)

# make results determinstic
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Customised Dataset class
class KMNIST(Dataset):
    
    def __init__(self,dataset):
        # Load the data from two consecutive timestamps of temperature 
        self.data = dataset
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        
        return data_item

temperature_dataset = KMNIST(
    temperature_fields
)

testingAndValidation_split = 0.2
validation_split = 0.1

# Creating data indices for training, testing and validation splits
# Reference: https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
temperature_dataset_size = len(temperature_dataset)
temperature_indices = list(range(temperature_dataset_size))

temperature_training_testing_split = int(np.floor(testingAndValidation_split * temperature_dataset_size))
temperature_testing_validation_split = int(np.floor(validation_split * temperature_dataset_size))

np.random.shuffle(temperature_indices)
temperature_train_indices, temperature_val_indices ,temperature_test_indices = temperature_indices[temperature_training_testing_split:], temperature_indices[:temperature_testing_validation_split], temperature_indices[temperature_testing_validation_split:temperature_training_testing_split] 

# Creating data samplers
temperature_train_sampler = SubsetRandomSampler(temperature_train_indices)
temperature_test_sampler = SubsetRandomSampler(temperature_test_indices)
temperature_valid_sampler = SubsetRandomSampler(temperature_val_indices)

train_loader = DataLoader(
    dataset=temperature_dataset,
    batch_size = batch_size,
    sampler=temperature_train_sampler,
)

test_loader = DataLoader(
    dataset=temperature_dataset,
    batch_size = batch_size,
    sampler=temperature_test_sampler,
)

validation_loader = DataLoader(
    dataset=temperature_dataset,
    batch_size = batch_size,
    sampler=temperature_valid_sampler,
)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :201, :]
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential( # 201x401 => 6x23x45
            nn.Conv2d(1, 3, stride=(3, 3), kernel_size=(5, 5), padding=2),
            nn.Tanh(),
            
            nn.Conv2d(3, 6, stride=(3, 3), kernel_size=(5, 5), padding=2),
            nn.Tanh(),
            
        )
        
    def forward(self, x):
        out=self.encoder(x)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential( # 6x23x45 => 201x401
            
            nn.ConvTranspose2d(6, 3, stride=(3, 3), kernel_size=(5, 5), padding=(2,2)),
            nn.Tanh(),
            
            nn.ConvTranspose2d(3, 1, stride=(3, 3), kernel_size=(5, 5), padding=(1,0)),
            
        )
        

    def forward(self, x):
        out=self.decoder(x)
        return out

    
def test(encoder, decoder, test_loader, device, color_regions):
    
    correct = 0
    total = 0
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    best_worst_error_list = [1000000, 0]
    best_worst_input_list = [0, 0]
    best_worst_predicted_list = [0, 0]
    
    with torch.no_grad():
        for data in test_loader:
            inputs = data.to(device)
            inputs = inputs.view(inputs.shape[0], 1, 201,401)
                
            # Get output features and calculate loss
            outputs = encoder(inputs)
            outputs = decoder(outputs)
            loss = criterion(outputs, inputs)
                
            # If the loss value is less than 0.01, we consider it being accurate
            for j in range(len(inputs)):
                single_loss = criterion(outputs[j], inputs[j])
                if single_loss.item() <= accurate_loss_baseline:
                    correct += 1
                total += 1
                
                # Record worst error
                if single_loss.item() > best_worst_error_list[1]:
                    best_worst_error_list[1] = single_loss.item()
                    best_worst_input_list[1] = inputs[j]
                    best_worst_predicted_list[1] = outputs[j]
                    
                # Record best error
                if single_loss.item() < best_worst_error_list[0]:
                    best_worst_error_list[0] = single_loss.item()
                    best_worst_input_list[0] = inputs[j]
                    best_worst_predicted_list[0] = outputs[j]
                    

            # Add to the validation loss
            total_loss += loss.item()
    
    text_file = open('ConvAE_testingData_Gadi_1.txt', "w")
    n1 = text_file.write(str(total_loss)+"\n")
    n2 = text_file.write(str(100*correct//total)+"\n")
    text_file.close()
    print("Testing Data saved! The path is ConvAE_testingData_Gadi_1.txt")
    
    
    text_file = open('ConvAE_testingData_Gadi_2.txt', "w")
    n1 = text_file.write("/".join([str(elem) for elem in best_worst_error_list])+"\n")
    n2 = text_file.write("/".join([str(elem.detach().numpy()[0].flatten()) for elem in best_worst_input_list])+"\n")
    n3 = text_file.write("/".join([str(elem.detach().numpy()[0].flatten()) for elem in best_worst_predicted_list]))         
    text_file.close()
    print("Testing Data saved! The path is ConvAE_testingData_Gadi_2.txt")

    
encoder = Encoder().to(device)
decoder = Decoder().to(device)
encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))

test(encoder, decoder, test_loader, device, 1)