import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        # Fully connected layers for hand input
        self.fc1 = nn.Linear(26, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Fully connected layers for noise input (optional)
        # self.noise_dim = 100
        # self.fc_noise = nn.Linear(self.noise_dim, 512)

        # Fully connected layer to expand to board size
        self.fc3 = nn.Linear(128, 50 * 50 * 256)
        
        # Reshape to match board dimensions and initial channels
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # Dilated convolutions for wide context
        self.dilated_conv1 = nn.Conv2d(32, 32, kernel_size=3, dilation=2, padding=2)
        self.dilated_conv2 = nn.Conv2d(32, 32, kernel_size=3, dilation=4, padding=4)

        # Final output layer
        self.output_conv = nn.Conv2d(32, 27, kernel_size=1)

    def forward(self, hand):
        # Process hand input through FC layers
        x = F.relu(self.fc1(hand))
        x = F.relu(self.fc2(x))

        # Generate noise and process through FC layer
        # noise = torch.randn(hand.size(0), self.noise_dim, device=hand.device)
        # noise = F.relu(self.fc_noise(noise))
        
        # Concatenate hand input and noise
        # x = torch.cat((x, noise), dim=1)

        # Expand to match board size
        x = F.relu(self.fc3(x))

        # Reshape to 4D tensor (batch_size, channels, height, width)
        x = x.view(-1, 256, 50, 50)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Dilated convolutions for capturing word context
        x = F.relu(self.dilated_conv1(x))
        x = F.relu(self.dilated_conv2(x))

        # Final output layer to produce board with probabilities over letters
        x = self.output_conv(x)
        
        # Apply softmax across the channel dimension to get probabilities
        x = F.softmax(x, dim=1)
        
        return x
