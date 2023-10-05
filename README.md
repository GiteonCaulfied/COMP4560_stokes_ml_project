# COMP4560_stokes_ml_project
A repository that we are going to use to keep track of project evolution, notes, ideas, etc.

I am going to use this repository as a sort of research diary with my own notes on the project.

## Geoid Problem

### Datasets
Datasets used for this problem are located in the folder `Data/Geoid/new_results_1k_zero`.

### Files
Files related to this problem includes:

```
Geoid_systematic_training.ipynb
Geoid_visualisation.ipynb
ModelList.txt
```

`Geoid_systematic_training.ipynb` is used for systematically training the models with their hyperparameters in the file `ModelList.txt` and saving them in the folder `1D_result`. 

`Geoid_visualisation.ipynb` is used for visualisation given a specific model's path.

## Mantle Convection Problem

### Supported by HPC

This problem is mainly researched with the help of a HPC system called Gadi. Files in the repository with a suffix of `.sh` are shell scripts submitted to Gadi in order to run the python script on Gadi. 


### Datasets
Datasets used for this problem can be found in the following URLs:

- [Limited Dataset](https://anu365-my.sharepoint.com/:f:/g/personal/u7189309_anu_edu_au/Em9tN9ofPRBBtJADs2G66rUBuY0WKp-2BEXNMI-U0a_JBw?e=pUFcF6)
- [Larger Dataset](https://anu365-my.sharepoint.com/:f:/g/personal/u7189309_anu_edu_au/EvC4GCemOlFKm1JxQl8pSbEBn6ORYK_hVNnXW5_J-fUOBg?e=j7uxc2)

Interpolated dataset is generated from the Larger dataset using `interpolation.py` and `interpolation_job.sh`.


### Convolutional Autoencoders (ConvAE)

Convolutional Autoencoder is used to compress the data before feeding the data into a predicting model.

Files related includes:

```
ConvAE_training.py
ConvAE_testing.py
ConvAE_training_job.sh
ConvAE_testing_job.sh
ConvAE_visualisation.ipynb
```

Training and testing results are stored in the folder `2D_ConvAE_results`


### Fully Connected Neural Network (FNN)

Fully Connected Neural Network is used to predict the temperature field at the next time step.

Files related includes:

```
FNN_training.py
FNN_testing.py
FNN_training_job.sh
FNN_testing_job.sh
FNN_visualisation.ipynb
```

Training and testing results are stored in the folder `2D_FNN_results`.


### Long Short-Term Memory (LSTM)

Long Short-Term Memory is used to predict rest of the temperature fields as a sequence given the first 50 temperature field in a simulation.

Files related includes:

```
LSTM_training.py
LSTM_testing.py
LSTM_training_job.sh
LSTM_testing_job.sh
LSTM_visualisation.ipynb
```

Training and testing results are stored in the folder `2D_LSTM_results`.
