# IMPORTS
# built-in libraries
import time
from datetime import datetime, timedelta
from pathlib import Path
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

parser = argparse.ArgumentParser(
    description="Trains nn models on the K-S system"
)
parser.add_argument('--epochs', required=False, type=int, default=200, help="How many training loops. Usually 0 or 200")
parser.add_argument('--batch_size', required=False, type=int, default=75, help="Number of rows of training data to sample before updating weights and biases")
parser.add_argument('--eqn', required=False, type=str, default='complete', help="What version of the KSE do you want to use (complete, nouux, nouxx, nouxxxx)?")
parser.add_argument('--act_func', required=False, type=str, default='sigmoid', help="Which activation function would you like to use in between model layers?")
parser.add_argument('--init_type', required=False, type=str, default='normal', help="Which initialization procedure you would like to use.")
parser.add_argument('--std', required=False, type=float, default=0.01, help="Standard deviation for initialization of weights.")
parser.add_argument('--space', required=True, type=str, default='real', help="The space of u in your KSE. Either 'real' or 'fourier'.")
parser.add_argument('--normed', required=False, type=bool, default=None, help="Whether or not to normalize the training data before processing")
parser.add_argument('--trial', required=False, type=int, default=1, help="The trial number of this training procedure.")
parser.add_argument('--save_epochs', required=False, type=int, nargs='*', default=None, help="The epochs when you want to save data.")
parser.add_argument('--resuming', required=False, type=bool, default=False, help="Are you continuing training from the most recent checkpoint?")
args = parser.parse_args()

# GLOBAL VARIABLES
# hyperparameters
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATES = [0.001, 0.0001] # the second entry isn't actually used in the code, its just there for reference

# model descriptors
TRAINER_VERSION = 9
TRAINING_SPACE = args.space
ACTIVATION_FUNC = args.act_func
INITIALIZATION_TYPE = args.init_type
INITIALIZATION_PARAMS = None
if INITIALIZATION_TYPE == 'normal':
    INITIALIZATION_PARAMS = [0.0, args.std]
EQUATION = args.eqn
NORMALIZED = args.normed
TRIAL_NUM = args.trial

if args.save_epochs is None:
    EPOCHS_TO_SAVE = np.concatenate((np.arange(1, 10, 1), np.arange(10, 30, 2), np.arange(30, 100, 5), np.arange(100, 200, 10), np.array([200])))
else:
    EPOCHS_TO_SAVE = args.save_epochs
RESUMING = args.resuming

# make file names from parameters
WEIGHTS_FOLDER = f"model_weights/"
WEIGHTS_FOLDER += f"trainer{TRAINER_VERSION}"
WEIGHTS_FOLDER += f"-{EQUATION}eq" if EQUATION != 'complete' else f""
WEIGHTS_FOLDER += f"-{ACTIVATION_FUNC}act" if ACTIVATION_FUNC != 'sigmoid' else f""
WEIGHTS_FOLDER += f"-{BATCH_SIZE}bt" if BATCH_SIZE != 75 else f""
WEIGHTS_FOLDER += f"-{INITIALIZATION_TYPE}init" if INITIALIZATION_TYPE != 'normal' else f""
WEIGHTS_FOLDER += f"-{INITIALIZATION_PARAMS[1]}std" if (INITIALIZATION_TYPE == 'normal' and INITIALIZATION_PARAMS[0] != 0) else f""
WEIGHTS_FOLDER += f"-{TRAINING_SPACE}sp"
WEIGHTS_FOLDER += f"-normed" if NORMALIZED else f""
WEIGHTS_FOLDER += f"-trial{TRIAL_NUM}"

TRAINING_DATA_FILE = f'training_data/'
TRAINING_DATA_FILE += f'u_hist' if TRAINING_SPACE == 'real' else f'uhat_hist'
TRAINING_DATA_FILE += f'_{EQUATION}' if EQUATION != 'complete' else f''
TRAINING_DATA_FILE += f'.h5'

# variable maps
series_length_map = {
    'complete': 40000,
    'nouux': 160,
    'nouxx': 48,
    'nouxxxx': 6
}
activation_map = {
    'sigmoid': nn.Sigmoid,
    'gelu': nn.GELU,
    'relu': nn.ReLU,
    'tanh': nn.Tanh
}
initialization_map = {
    'normal': 0,
    'kaiming_normal': 1,
    'orthogonal': 2
}

# where to do the processing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DATASET PREPROCESSING
class HDFDataset(Dataset):
    def __init__(self, hdf_file, series_length = 40000):
        # convert hdf5 data to a pytorch tensor
        df = pd.read_hdf(hdf_file, key='df')
        self.raw_data = torch.tensor(df.values, dtype=torch.float32) 
        self.data = self.raw_data

        self.series_length = series_length
        # the number of time series in the training data
        self.num_series = self.data.shape[0] // self.series_length
        # -1 because the last row doesn't have a next row to be with
        self.samples_per_series = self.series_length - 1
    
    def __len__(self):
        # how many time steps of data, minus the last time step of each time series
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

    @property
    def num_features(self):
        # number of input/output features for the nn
        return self.data.size(dim=1)
    
    def normalize(self, mean=None, std=None):
        # normalize the data in each column
        if mean is None:
            mean = self.raw_data.mean(dim=0)
        if std is None:
            std = self.raw_data.std(dim=0)
        self.mean = mean
        self.std = std
        self.data = (self.raw_data - self.mean) / self.std

# DEFINE MODEL ARCHITECTURE
class neuralODE(nn.Module):
    def __init__(self, act_func='sigmoid', dim_inout=64, init_type='normal', init_params=[0.0, 0.01]):
        super(neuralODE, self).__init__()

        # define the string and nn module versions of the activation function
        self.act_func = act_func
        self.act_layer = activation_map[self.act_func]

        # do stuff with the model initialization parameters so that they can be saved to the register buffer later
        self.init_record = torch.tensor([initialization_map[init_type]])
        if self.init_record.item() == 0:
            self.init_record = torch.cat((self.init_record, torch.tensor(init_params)))

        # save model parameters to the register buffer for access during model evaluation/ plotting
        self.register_buffer('layer_init_record', self.init_record)
        self.register_buffer('data_mean', torch.zeros(dim_inout)) # placeholder mean
        self.register_buffer('data_std', torch.ones(dim_inout)) # placeholder stdev

        # use a linear & sigmoid fully connected nn
        self.net = nn.Sequential(
            nn.Linear(dim_inout, 200),
            self.act_layer(),
            nn.Linear(200, 200),
            self.act_layer(),
            nn.Linear(200, 200),
            self.act_layer(),
            nn.Linear(200, dim_inout)
        )
        self.init_weights()

    # if training data is normalized, save the real mean and standard deviation of the dataset
    def set_normalization_stats(self, mean, std):
        self.data_mean = mean
        self.data_std = std

    # initialize model weights & biases
    def init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                if self.init_record[0] == 0:
                    # initialization for random normal weights
                    nn.init.normal_(m.weight, mean=self.init_record[1], std=self.init_record[2])
                elif self.init_record[0] == 1:
                    # initialization for kaiming normal weights
                    nn.init.kaiming_normal_(m.weight, nonlinearity=self.act_func)
                elif self.init_record[0] == 2:
                    # initialization for orthogonal weights
                    nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    # just your standard everyday average forward pass function with a t variable for integration
    def forward(self, t, x):
        return self.net(x)

# MODEL SET UP
def setup_model(hdf_file, equation, batch_size, init_type, init_params=None):
    # load dataset and normalize if applicable
    dataset = HDFDataset(hdf_file=hdf_file, series_length=series_length_map[equation])
    if NORMALIZED: 
        dataset.normalize()

    # put the data into a train laoder
    train_loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        num_workers=0, # set to 0 because training on a windows device
        shuffle=True
    )

    # initialize the model
    model = neuralODE(dim_inout=dataset.num_features, init_type=init_type, init_params=init_params).to(device)
    if NORMALIZED: 
        model.set_normalization_stats(dataset.mean, dataset.std)

    return model, train_loader

# TRAIN MODEL
def train_model(model, learning_rates, resuming, epochs, train_loader, epochs_to_save, weights_folder):
    # training setup
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rates[0])
    scheduler = MultiStepLR(optimizer, milestones=[epochs // 2], gamma=0.1)
    integration_time = torch.tensor([0.0, 0.25]).to(device)

    # load latest checkpoint if specified in argument
    if resuming:
        checkpoint = torch.load('model_weights/latest-checkpoint.pth', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        prev_epoch = 0

    # record training start time
    start_runtime = time.perf_counter()

    # training loop
    print("Beginning training...")
    for epoch in range(prev_epoch, epochs):
        epoch_loss = 0.0
        model.train()

        # per-epoch timer (seconds)
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
            }, 'model_weights/latest-checkpoint.pth')

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

# SAVE MODEL WEIGHTS
def save_weights(weights_folder, epoch, model):
    weights_dir = Path(weights_folder)
    if not weights_dir.exists(): weights_dir.mkdir(parents=True, exist_ok=False)
    weights_path = Path(weights_folder + f"/epoch{epoch:03d}.pth")

    torch.save(model.state_dict(), weights_path)
    print(f"Model weights successfully saved to {weights_path}")

# VALIDATE GLOBAL VARIABLES/ PARAMETERS
def validate_vars(equation, training_space, initialization_type, act_func, weights_folder, resuming):
    if equation not in series_length_map:
        raise ValueError("Unsupported equation. Must use 'complete', 'nouux', 'nouxx', or 'nouxxxx'.")
    if training_space not in ['real', 'fourier']:
        raise ValueError("Unsupported training space. Must use either real or fourier")
    if initialization_type not in ['normal', 'kaiming_normal', 'orthogonal']:
        raise ValueError("Unsupported initialization procedure. You must use either normal, kaiming_normal, or orthogonal.")
    if act_func not in activation_map:
            raise ValueError(f"Unsupported activation: {act_func}")
    if Path(weights_folder).exists() and not resuming:
        raise FileExistsError(f"A folder with the name {weights_folder} already exists. Please increment the trial number and try again.")
    

if __name__ == '__main__':
    validate_vars(EQUATION, TRAINING_SPACE, INITIALIZATION_TYPE, ACTIVATION_FUNC, WEIGHTS_FOLDER, RESUMING)

    modela, train_loadera = setup_model(TRAINING_DATA_FILE, EQUATION, BATCH_SIZE, INITIALIZATION_TYPE, INITIALIZATION_PARAMS)
    train_model(modela, LEARNING_RATES, RESUMING, EPOCHS, train_loadera, EPOCHS_TO_SAVE, WEIGHTS_FOLDER)