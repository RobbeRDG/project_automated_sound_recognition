from torch import nn


class BaselineModel(nn.Module):
    def __init__(self, num_in_channels, num_out_labels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                    in_channels= num_in_channels,
                    out_channels= 64,
                    kernel_size= 5,
                    stride= 1
                ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(
                    kernel_size= (1,4),
                    stride= 1
                ),
            nn.Dropout(0.3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                    in_channels= 64,
                    out_channels= 32,
                    kernel_size= 5,
                    stride= 1
                ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(
                    kernel_size= (1,4),
                    stride= 1
                ),
            nn.Dropout(0.3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                    in_channels= 32,
                    out_channels= 16,
                    kernel_size= 5,
                    stride= 1
                ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(
                    kernel_size= (1,4),
                    stride= 1
                ),
            nn.Dropout(0.3)
        )

        self.flatten = nn.Flatten()

        self.dense1 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(100, num_out_labels),
            nn.ReLU()
        )

        self.softmax = nn.Softmax(dim= 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        y = self.softmax(x)

        return y