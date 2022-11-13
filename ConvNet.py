import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode, image_size, num_classes):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        # Pool over 2x2 regions, 40 kernels, stride =1, with kernel size of 5x5.
        # define first conv laver 
        
        
        # Fully connected layers
        # define the layers according to the model
        if mode == 1:
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(image_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 784),
                nn.Sigmoid()
            )

        if mode == 2:
            # Encoder
            self.encoder = nn.Sequential(
                # input.shape = [1, 28, 28]
                nn.Conv2d(1, 16, 3, stride=(1, 1), padding=(1, 1)), # in_channels, out_channels, kernel_size
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                # output.shape = [16, 14, 14]
                nn.Conv2d(16, 32, 3,stride=(1, 1), padding=(1,1)),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                # output.shape = [32, 7, 7]
            )

            self.decoder = nn.Sequential(
                # output.shape = [32, 7, 7]
                nn.ConvTranspose2d(32, 16, 3,stride=(2, 2), padding=(1,1), output_padding=1), # in_channels, out_channels, kernel_size
                # output.shape = [16, 14, 14]
                nn.ConvTranspose2d(16, 8, 3,stride=(2, 2), padding=(1,1), output_padding=1),
                # output.shape = [8, 28, 28]
                nn.UpsamplingBilinear2d(scale_factor=2),
                # output.shape = [8, 56, 56]
                nn.Conv2d(8, 1, 3,stride=(2, 2), padding=(1,1)),
                # output.shape = [1, 28, 28]
                nn.Sigmoid()
            )


        self.forward = self.model
        
        
    def model(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)

        return decoded
