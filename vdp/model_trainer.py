# IMPORTS
# built-in libraries
import time
from datetime import datetime, timedelta
from pathlib import Path
from warnings import warn
import argparse

# non-ml libraries
import numpy as np
import pandas as pd

# ml related libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchdiffeq import odeint

parser = argparse.ArgumentParser(description="Trains nn models on the Van Der Pol system")
parser.add_argument('--batch_size', required=False, type=int, default=75, help="Number of rows of training data to sample before updating weights and biases")
parser.add_argument('--epochs', required=False, type=int, default=200, help="How many training loops. Usually 0 or 200")
parser.add_argument('--eqn', required=False, type=str, default='complete', help="What version of the Van Der Pol Equation do you want to use (complete, damped)?")
parser.add_argument('--data', required=True, type=str, default='all', help="What regions of the phase portrait is your training data from (inside, outside, all)")
parser.add_argument('--std', required=False, type=float, default=0.01, help="Standard deviation for initialization of weights.")
parser.add_argument('--normed', required=False, type=float, default=False, help="Whether or not to normalize the training data before processing")
parser.add_argument('--space', required=False, type=str, default='real', help="The space of training data you will be using. Either 'real' or 'fourier'.")
parser.add_argument('--trial', required=False, type=int, default=1, help="The trial number of this training procedure.")
parser.add_argument('--resuming', required=False, type=bool, default=False, help="Are you continuing training from the most recent checkpoint?")
args = parser.parse_args()

# GLOBAL VARIABLES
# hyperparameters
BATCH_SIZE = args.batch_size # default = 75; tried
LEARNING_RATES = [0.001, 0.0001] # the second entry isn't actually used in the code, its just there for reference
EPOCHS = args.epochs # usually 0 or 200
EPOCHS_TO_SAVE = np.concatenate((np.arange(1, 10, 1), np.arange(10, 30, 2), np.arange(30, 100, 5), np.arange(100, max(110,EPOCHS), 10), np.array([EPOCHS])))

# model metadata
TRAINER_VERSION = 8
EQUATION = args.eqn
OBSERVABLE_DATA = args.data
INIT_STD = args.std
NORMALIZED = args.normed
TRAINING_SPACE = args.space
TRIAL_NUM = args.trial
RESUMING = args.resuming

WEIGHTS_FOLDER = f"../model_weights/trainer{TRAINER_VERSION}"
WEIGHTS_FOLDER += f"-{EQUATION}eq" if EQUATION != 'complete' else f""
WEIGHTS_FOLDER += f"-{OBSERVABLE_DATA}data"
WEIGHTS_FOLDER += f"-{INIT_STD}std" if INIT_STD != 0.01 else f""
WEIGHTS_FOLDER += f"-normalized" if NORMALIZED else f""
WEIGHTS_FOLDER += f"-t{TRIAL_NUM}"

# variable maps 
file_map = {
    'complete': {
        'inside': '../data/uhist-inside.hdf5',
        'outside': '../data/uhist-outside.hdf5',
        'all': '../data/uhist-all.hdf5'
    },
    'damped': {
        'inside': None,
        'outside': None,
        'all': '../data/uhist-damped-all.hdf5'
    }
}
series_length_map = {
    'complete': 400,
    'damped': 400
}

# where to do the processing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DATASET PREPROCESSING
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, series_length = 40000):
        # convert csv data to a pytorch tensor
        df = pd.read_hdf(hdf5_file, key='df')
        self.raw_data = torch.tensor(df.values, dtype=torch.float32) 
        self.data = self.raw_data

        self.mean = self.raw_data.mean(dim=0)
        self.std = self.raw_data.std(dim=0)

        self.series_length = series_length
        # the number of time series in the training data
        self.num_series = self.data.shape[0] // self.series_length
        # -1 because the last row doesn't have a next row to be with
        self.samples_per_series = self.series_length - 1
    
    def __len__(self):
        return self.num_series * self.samples_per_series
    
    def __getitem__(self, idx):
        # some weird math to skip over the last row of each time series when indexing through the dataset
        series_idx = idx // self.samples_per_series
        local_step_idx = idx % self.samples_per_series
        current_row_idx = (series_idx * self.series_length) + local_step_idx

        # return the current row and the next row for one-step prediction
        current_row = self.data[current_row_idx]
        next_row = self.data[current_row_idx + 1]
        return current_row, next_row

    def normalize(self):
        self.data = (self.raw_data - self.mean) / self.std

# DEFINE MODEL ARCHITECTURE
class neuralODE(nn.Module):
    def __init__(self, act_layer=nn.Sigmoid, dim_inout=2, init_std=0.01):
        super(neuralODE, self).__init__()

        # save the initial standard deviation to the reigster buffer
        self.register_buffer('init_std', torch.tensor([init_std]))

        # save space in the register buffer for data means and standard deviations (1 per column)
        self.register_buffer('data_mean', torch.zeros(dim_inout)) # placeholder mean
        self.register_buffer('data_std', torch.ones(dim_inout)) # placeholder stdev

        # use a linear & sigmoid fully connected nn
        self.net = nn.Sequential(
            nn.Linear(dim_inout, 200),
            act_layer(),
            nn.Linear(200, 200),
            act_layer(),
            nn.Linear(200, 200),
            act_layer(),
            nn.Linear(200, dim_inout)
        )
        self.init_weights()

    # initialize model weights & biases
    def init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # initialize based on inputted stdev
                nn.init.normal_(m.weight, mean=0, std=self.init_std.item())
                nn.init.constant_(m.bias, 0.0)

    # just your standard everyday average forward pass function with a t variable for integration
    def forward(self, t, x):
        return self.net(x)

    # sets the data_mean and data_std in the register buffer
    def set_normalization_stats(self, mean, std):
        self.data_mean = mean
        self.data_std = std

# MODEL SET UP
def setup_model(hdf5_file, equation, batch_size, init_std=0.01):
    # load dataset and normalize if applicable
    dataset = HDF5Dataset(hdf5_file=hdf5_file, series_length=series_length_map[equation])

    # put the data into a train laoder
    train_loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        num_workers=0, 
        shuffle=True
    )

    # initialize the model
    model = neuralODE(init_std=init_std).to(device)
    model.set_normalization_stats(dataset.mean, dataset.std)

    return model, train_loader

def train_model(model, learning_rates, resuming, epochs, train_loader, epochs_to_save, weights_folder):
    # training setup
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rates[0])
    scheduler = MultiStepLR(optimizer, milestones=[epochs // 2], gamma=0.1)
    integration_time = torch.tensor([0.0, 0.05]).to(device)

    # load latest checkpoint if specified in argument
    if resuming:
        checkpoint = torch.load('../model_weights/latest-checkpoint.pth', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        prev_epoch = 0

    # start timer
    start_runtime = time.perf_counter()

    # training loop
    print("Beginning training...")
    for epoch in range(prev_epoch, epochs):
        epoch_loss = 0.0
        model.train()

        # per-epoch timer
        epoch_start = time.perf_counter()
        
        for _, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            predictions = odeint(model, batch_x, integration_time, method='dopri5')
            y_next_pred = predictions[1]
            loss = loss_fn(y_next_pred, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # keep track of loss
            epoch_loss += loss.item()

        scheduler.step()
        
        # save full checkpoint
        torch.save({
            'epoch': epoch+1, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(), 
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss, 
            }, '../model_weights/latest-checkpoint.pth')

        # save model weights at specified epochs for evaluation
        if epoch+1 in epochs_to_save:
            save_weights(weights_folder, epoch+1, model)

        # progress print statement
        epoch_time = time.perf_counter() - epoch_start
        avg_loss = epoch_loss / len(train_loader) # average batch MAE loss
        runtime = time.perf_counter() - start_runtime
        mins_remaining = ((runtime/((epoch+1) - prev_epoch)) * (epochs-(epoch+1)))/60
        est_completion = datetime.now() + timedelta(minutes=mins_remaining)
        print(f"Epoch [{epoch+1}/{epochs}], Avg Batch Loss: {avg_loss:.9f}, Epoch time: {epoch_time:.2f}s, Est Completion: {est_completion:%H:%M:%S}")

def save_weights(weights_folder, epoch, model):
    if not Path(weights_folder).exists():
        Path(weights_folder).mkdir()

    weights_path = f"{weights_folder}/epoch{epoch:03d}.pth"

    torch.save(model.state_dict(), weights_path)
    print(f"Model weights successfully saved to {weights_path}")

def validate_vars(equation, observable_data, weights_folder, resuming):
    """validate training variables"""
    if equation not in ['complete', 'damped']:
        raise ValueError("Unsupported equation. Must use 'complete' or 'damped'.")
    if observable_data not in ['inside', 'outside', 'all']:
        raise ValueError("Unsupported observable data argument. Must be inside, outside, or all.")
    if (Path(weights_folder).exists() or Path(weights_folder + '.pth').exists()) and not resuming:
        raise FileExistsError(f"A folder or file with the name {weights_folder} already exists. Please increment the trial number and try again.")
    

if __name__ == '__main__':
    validate_vars(EQUATION, OBSERVABLE_DATA, WEIGHTS_FOLDER, RESUMING)

    modela, train_loadera = setup_model(file_map[EQUATION][OBSERVABLE_DATA], EQUATION, BATCH_SIZE, INIT_STD)
    train_model(modela, LEARNING_RATES, RESUMING, EPOCHS, train_loadera, EPOCHS_TO_SAVE, WEIGHTS_FOLDER)