import sys
import os

import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import util


class FeatureExtrator(nn.Module):
    def __init__(self, state_shape):
        super().__init__()

        channel, height, width = state_shape
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            # nn.ReLU(),
            nn.Flatten(),
        )

        height, width = util.shape_after_conv2d(height, width, kernel_size=(8, 8), stride=(4, 4))
        height, width = util.shape_after_conv2d(height, width, kernel_size=(4, 4), stride=(2, 2))
        height, width = util.shape_after_conv2d(height, width, kernel_size=(3, 3), stride=(1, 1))
        self.output_size = 64 * height * width

    def forward(self, x):
        return self.network(x)

class InverseModel(nn.Module):
    def __init__(self, feature_size2, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size2, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.network(x)

class ForwardModel(nn.Module):
    def __init__(self, feature_action_size, feature_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_action_size, 512),
            nn.ReLU(),
            nn.Linear(512, feature_size),
        )

    def forward(self, x):
        return self.network(x)
