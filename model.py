import torch
from torch import nn
from torchsummary import summary
import utils
import os


"""
CNN with 3 convolutional blocks (each composed by 2 2d-convolutions, ELU activation function, He_uniform initializer, 
padding same, kernels size equal to 4-8-16), flatten, first dense layer with 10 neurons and the last with 2 classes).
"""

if torch.cuda.is_available():
    device = torch.device("cuda:6")
else:
    device = torch.device("cpu")


class SirenCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=(1, 1)
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=4,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=(1, 1)
            ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=(1, 1)
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=8,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=(1, 1)
            ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=(1, 1)
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=(1, 1)
            ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=16 * 8 * 4 * 3, out_features=10),
            nn.ELU(),
            nn.Linear(in_features=10, out_features=2),
        )
        self.softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_data):
        x = self.conv1(input_data.to(device))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        logits = self.dense_layers(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":

    # Device definition
    if torch.cuda.is_available():
        device = torch.device("cuda:6")
        print("Device: {}".format(device))
        print("Device name: {}".format(torch.cuda.get_device_properties(device).name))
    else:
        device = torch.device("cpu")
        print("Device: {}".format(device))

    # Print CNN architecture
    cnn = SirenCNN()
    summary(cnn.to(device), (1, 128, 51))

    # Load the best saved model
    checkpoint = os.path.join("./models/", "model_best.pt")
    saved_model = utils.load_checkpoint(checkpoint, cnn, optimizer=None, parallel=False)