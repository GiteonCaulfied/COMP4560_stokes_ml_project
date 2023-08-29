import h5py
import os

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler


# Paths
path = "/scratch/kr97/xh7958/comp4560/solutions"
encoder_path = "Conv2D_encoder_best_Gadi.pth"
decoder_path = "Conv2D_decoder_best_Gadi.pth"

#Parameters
n_epoch = 300
batch_size = 16
lr = 5e-5
betas = (0.9, 0.999)

accurate_loss_baseline = 1e-6


# Temperature for the two consecutive timestamp
temperature_fields_input = []
temperature_fields_output = []

# Read text File  
def read_text_file(file_path):
    with h5py.File(file_path, 'r') as f:
        temperature_fields_input.append(f['temperature'][:][:-1])
        temperature_fields_output.append(f['temperature'][:][1:])
        
        
# Iterate through all file
for file in os.listdir(path):
    file_path = f"{path}/{file}"
  
    # call read text file function
    read_text_file(file_path)
    #print(f"{file_path} is finished reading")

temperature_fields_input = np.asarray(temperature_fields_input).reshape(9900,201,401)
temperature_fields_output = np.asarray(temperature_fields_output).reshape(9900,201,401)


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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential( # 1x201x401 => 6x23x45
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
    
encoder = Encoder().to(device)
decoder = Decoder().to(device)
encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))

print("Encoder and Decoder loaded!")


# Customised Dataset class
class KMNIST(Dataset):
    
    def __init__(self, input_data, output_data):
        self.input = input_data
        self.output = output_data
        
    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_item = self.input[index]
        output_item = self.output[index]
        
        return input_item, output_item
    
temperature_dataset = KMNIST(
    temperature_fields_input,
    temperature_fields_output
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()    
        self.fc1 = nn.Linear(6210,3105)
        self.fc2 = nn.Linear(3105,1035)
        self.fc3 = nn.Linear(1035,3105)
        self.fc4 = nn.Linear(3105,6210)
        self.act = nn.Tanh()
        
        '''
        self.fc1 = nn.Linear(6210,1000)
        self.fc2 = nn.Linear(1000,200)
        self.fc3 = nn.Linear(200,1000)
        self.fc4 = nn.Linear(1000,6210)
        '''
    
    def forward(self, x):
        
        output_batch_size = x.shape[0]
        x = x.view(-1, 6210)
        
        out = self.fc1(x)
        out = self.act(out)
        
        out = self.fc2(out)
        out = self.act(out)
        
        out = self.fc3(out)
        out = self.act(out)
        
        out = self.fc4(out)
        
        out = out.view(output_batch_size,6,23,45)
        
        return out
    

    
def train(model, encoder, train_loader, val_loader, device, optimizer, n_epoch):
    
    criterion = nn.MSELoss()
    
    minimum_validation_loss = 10000000
    best_model_index = -1
    
    running_loss_list = []
    validation_loss_list = []

    # n_epoch times of iterations
    for epoch in range(n_epoch):

        running_loss = 0.0

        model.train()
        
        for data in train_loader:
            # get a batch of inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.view(inputs.shape[0], 1, 201, 401)
            labels = labels.view(labels.shape[0], 1, 201, 401)

            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            # Get output features, calculate loss and optimize
            outputs = encoder(inputs)
            labels = encoder(labels)
            
            outputs = model(outputs)
            loss = criterion(outputs.float(), labels.float())
            
            loss.backward()
            optimizer.step()

            # Add to the total training loss
            running_loss += loss.item()

        # print some statistics
        print(epoch+1,"epochs have finished")
        print("Current training loss is ",running_loss)
        running_loss_list.append(running_loss)
        running_loss = 0.0

        # Valiadation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            valid_loss = 0.0
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = inputs.view(inputs.shape[0], 1, 201, 401)
                labels = labels.view(labels.shape[0], 1, 201, 401)
                
                # Get output features and calculate loss
                outputs = encoder(inputs)
                labels = encoder(labels)
                
                outputs = model(outputs)
                loss = criterion(outputs, labels)
                
                # If the loss value is less than 0.01, we consider it being accurate
                for j in range(len(labels)):
                    single_loss = criterion(outputs[j], labels[j])
                    if single_loss.item() <= 0.01:
                        correct += 1
                    total += 1

                # Add to the validation loss
                valid_loss += loss.item()

            # Calculate valiadation accuracy and print Validation statistics
            print("Validation loss for this epoch is",valid_loss)
            print("Validation Accuracy for this epoch is", 100*correct//total)
            validation_loss_list.append(valid_loss)

        # Update the statistics for the best model
        if valid_loss <= minimum_validation_loss:
            minimum_validation_loss = valid_loss

            # Store the best models
        
            PATH = 'latent_space_model_best_Gadi.pth'

            torch.save(model.state_dict(), PATH)
            print("This model is now saved to Path:",PATH)
            
            best_model_index = epoch
            
        print()
    
    # Training finished, print the statistics for the best model
    print('Finished Training')
    print("Best model has a validation loss of ",minimum_validation_loss)
    print("Best model is in epoch ",best_model_index+1)
    
    # Save the Training loss and validation loss to a file
    text_file = open('LatentSpace_Prediction_trainingData_Gadi.txt', "w")
    n1 = text_file.write("/".join([str(elem) for elem in running_loss_list])+"\n")
    n2 = text_file.write("/".join([str(elem) for elem in validation_loss_list]))
    text_file.close()
    print("Training Data saved! The path is LatentSpace_Prediction_trainingData_Gadi.txt")
    

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)

train(model, encoder, train_loader, validation_loader, device, optimizer, n_epoch)