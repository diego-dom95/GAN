import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample
import h5py
import random
from waveform_utils import WaveformData
import matplotlib.pyplot as plt

# Define the custom dataset for loading waveform data
class WaveformDataset(Dataset):
    def __init__(self, folder_path, target_length=None, augment=False):
        """
        Custom Dataset for WaveformData

        Parameters:
        - folder_path: str, path to the directory containing waveform h5 files.
        - target_length: int, optional. Target length to resample all waveforms to.
        - augment: bool, optional. Whether to apply data augmentation to waveforms.
        """
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
        self.target_length = target_length
        self.augment = augment

        # If target length is not provided, determine it as the median length of all waveforms
        if target_length is None:
            lengths = [WaveformData(file_path=os.path.join(folder_path, file)).hp.shape[0] for file in self.files]
            self.target_length = int(np.median(lengths))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.files[idx])
        
        # Load waveform data using WaveformData class
        waveform = WaveformData(file_path = file_path)
        norm_params = waveform.norm_params
        hp = waveform.hp.numpy()
        hc = waveform.hc.numpy()
        params = waveform.params
        original_length = len(hp)

        # Resample waveforms to the target length
        if len(hp) != self.target_length:
            hp = resample(hp, self.target_length)
            hc = resample(hc, self.target_length)

        # Apply data augmentation if enabled
        if self.augment:
            if random.random() > 0.5:
                # Add Gaussian noise
                noise = np.random.normal(0, 0.01, hp.shape)
                hp += noise
                hc += noise
            if random.random() > 0.5:
                # Scale amplitude
                scale_factor = random.uniform(0.9, 1.1)
                hp *= scale_factor
                hc *= scale_factor

        # Convert to torch tensors
        hp_tensor = torch.tensor(hp, dtype=torch.float32)
        hc_tensor = torch.tensor(hc, dtype=torch.float32)
        params_tensor = torch.tensor([params['q'], params['spin1'], params['spin2']], dtype=torch.float32)
        original_length_tensor = torch.tensor(original_length, dtype=torch.float32)

        return params_tensor, hp_tensor, hc_tensor, norm_params, original_length_tensor

#============================================================
#                        GAN
#============================================================

# Generator Network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, x):
        return self.network(x)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output a probability
        )

    def forward(self, x):
        return self.network(x)

# GAN Class
class GAN:
    def __init__(self, input_dim, output_dim, learning_rate=0.0002):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(input_dim=6, output_dim=output_dim).to(self.device)  # Adding 3 for q, spin1, spin2
        self.discriminator = Discriminator(input_dim=output_dim).to(self.device)

        # Optimizers
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        # Loss function
        self.criterion = nn.BCELoss()
        self.losses_g = []
        self.losses_d = []

    def train(self, train_loader, val_loader, num_epochs=1000, noise_dim=3):
        for epoch in range(num_epochs):
            # Training phase
            for params, hp, _, _, _ in train_loader:
                params, hp = params.to(self.device), hp.to(self.device)

                # Prepare real data
                real_waveforms = hp.view(len(hp), -1)
                real_labels = torch.ones(len(hp), 1).to(self.device)

                # Generate fake waveforms
                noise = torch.randn(len(hp), noise_dim).to(self.device)
                generator_input = torch.cat((params, noise), dim=1)  # Concatenate params with noise
                fake_waveforms = self.generator(generator_input)
                fake_labels = torch.zeros(len(hp), 1).to(self.device)

                # Train Discriminator
                self.optimizer_d.zero_grad()
                output_real = self.discriminator(real_waveforms)
                loss_real = self.criterion(output_real, real_labels)
                output_fake = self.discriminator(fake_waveforms.detach())
                loss_fake = self.criterion(output_fake, fake_labels)
                loss_d = (loss_real + loss_fake) / 2
                loss_d.backward()
                self.optimizer_d.step()

                # Train Generator
                self.optimizer_g.zero_grad()
                output_fake = self.discriminator(fake_waveforms)
                loss_g = self.criterion(output_fake, real_labels)  # We want the discriminator to think the fakes are real
                loss_g.backward()
                self.optimizer_g.step()

            # Store losses
            self.losses_d.append(loss_d.item())
            self.losses_g.append(loss_g.item())

            # Print loss every epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

        # Plot the training losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses_d, label='Discriminator Loss')
        plt.plot(self.losses_g, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Training Losses')
        plt.show()

    def save_model(self, model_name):
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/GAN_models', exist_ok=True)
        generator_path = os.path.join('models/GAN_models', f"{model_name}_generator.pth")
        discriminator_path = os.path.join('models/GAN_models', f"{model_name}_discriminator.pth")
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)
        print(f"Models saved: {generator_path}, {discriminator_path}")

#============================================================
#                     LENGTH ESTIMATOR
#============================================================

# Length Estimator Network
class LengthEstimator(nn.Module):
    def __init__(self):
        super(LengthEstimator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Output the estimated length
            nn.Softplus()  # Ensure the output is positive
        )
        self.losses = []

    def forward(self, x):
        return self.network(x)

    def train_length_estimator(self, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        for epoch in range(num_epochs):
            # Training phase
            for params, _, _, _, original_length in train_loader:
                params = params.to(self.device)
                original_length = original_length.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                length_pred = self(params)
                loss = criterion(length_pred.squeeze(), original_length)
                loss.backward()
                optimizer.step()

            # Store loss
            self.losses.append(loss.item())

            # Print loss every epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], Length Estimator Loss: {loss.item()}")

        # Plot the training losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='Length Estimator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Length Estimator Training Loss')
        plt.show()

    def save_model(self, model_name):
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/LENGTH_model', exist_ok=True)
        model_path = os.path.join('models/LENGTH_model', f"{model_name}_length_estimator.pth")
        torch.save(self.state_dict(), model_path)
        print(f"Length estimator model saved: {model_path}")

# After GAN generates the waveform, estimate the length and resample
def recover_original_time_scale(generated_waveform, params, length_estimator, device):
    """
    Resample the generated waveform to match the estimated original length.
    
    Parameters:
    - generated_waveform: torch.Tensor, the generated waveform from the GAN.
    - params: torch.Tensor, the input parameters (q, spin1, spin2).
    - length_estimator: LengthEstimator, the trained length estimator model.
    - device: torch.device, device to run calculations on.
    
    Returns:
    - resampled_waveform: numpy array, the resampled waveform with estimated original length.
    """
    # Estimate the original length from the parameters
    params = params.to(device)
    estimated_length = length_estimator(params).item()
    estimated_length = int(estimated_length)  # Convert to integer
    
    # Resample the generated waveform to the estimated length
    generated_waveform = generated_waveform.detach().cpu().numpy()
    resampled_waveform = resample(generated_waveform, estimated_length)
    
    return resampled_waveform

#===============================================
#                   CNN
#===============================================

# CNN Network
class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * (input_dim // 4), 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.losses = []

    def forward(self, x):
        return self.network(x)

    def train_cnn(self, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        for epoch in range(num_epochs):
            # Training phase
            for params, hp, _, _, _ in train_loader:
                params, hp = params.to(self.device), hp.to(self.device)
                hp = hp.unsqueeze(1)  # Add channel dimension for CNN

                # Forward pass
                optimizer.zero_grad()
                output = self(params)
                loss = criterion(output, hp)
                loss.backward()
                optimizer.step()

            # Store loss
            self.losses.append(loss.item())

            # Print loss every epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], CNN Loss: {loss.item()}")

        # Plot the training losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='CNN Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('CNN Training Loss')
        plt.show()

    def save_model(self, model_name):
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/CNN_models', exist_ok=True)
        model_path = os.path.join('models/CNN_models', f"{model_name}_cnn.pth")
        torch.save(self.state_dict(), model_path)
        print(f"CNN model saved: {model_path}")

#=======================================================
#                   MULTI-LAYER PERCEPTRON
#=======================================================

# MLP Network
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.losses = []

    def forward(self, x):
        return self.network(x)

    def train_mlp(self, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        for epoch in range(num_epochs):
            # Training phase
            for params, hp, _, _, _ in train_loader:
                params, hp = params.to(self.device), hp.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                output = self(params)
                loss = criterion(output, hp)
                loss.backward()
                optimizer.step()

            # Store loss
            self.losses.append(loss.item())

            # Print loss every epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], MLP Loss: {loss.item()}")

        # Plot the training losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='MLP Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('MLP Training Loss')
        plt.show()

    def save_model(self, model_name):
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/MLP_models', exist_ok=True)
        model_path = os.path.join('models/MLP_models', f"{model_name}_mlp.pth")
        torch.save(self.state_dict(), model_path)
        print(f"MLP model saved: {model_path}")
