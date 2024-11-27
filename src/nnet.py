"""
Neural Network for the game.
"""

import torch
from torch import nn
import torch.nn.functional as F

from game import Game


class NNet(nn.Module):
    """
    A Convolutional Neural Network for the game, designed to process board state
    and predict action probabilities and state value.
    """

    def __init__(self, game: Game, args):
        super().__init__()

        # Game parameters
        board_size, var_size = game.get_input_size()
        self.board_x, self.board_y, self.board_z = board_size
        self.var_size = var_size
        self.action_size = game.get_action_size()
        self.args = args

        # CNN for image input
        self.conv1 = nn.Conv2d(
            self.board_x, args.num_channels, kernel_size=5, stride=2, padding=2
        )
        self.conv2 = nn.Conv2d(
            args.num_channels, args.num_channels, kernel_size=5, stride=2, padding=2
        )
        self.conv3 = nn.Conv2d(
            args.num_channels, args.num_channels, kernel_size=5, stride=2, padding=2
        )

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)

        # Calculate dynamic output size after convolutions
        final_board_x, final_board_y = self.calculate_conv_output(
            self.board_y, self.board_z
        )

        # Fully connected layers for image features
        self.fc_img1 = nn.Linear(args.num_channels * final_board_x * final_board_y, 512)
        self.fc_img2 = nn.Linear(512, 256)

        # Fully connected layers for variables
        self.fc_var1 = nn.Linear(var_size, 32)
        self.fc_var2 = nn.Linear(32, 32)

        # Combined fully connected layers
        self.fc_combined1 = nn.Linear(256 + 32, 256)
        self.dropout_combined1 = nn.Dropout(args.dropout)  # Dropout added here
        self.fc_combined2 = nn.Linear(256, self.action_size)
        self.fc_value = nn.Linear(256, 1)

    def calculate_conv_output(self, height, width):
        """
        Dynamically calculates the output size after 3 Conv2d layers.
        """
        for conv in [self.conv1, self.conv2, self.conv3]:
            height = (
                height + 2 * conv.padding[0] - conv.kernel_size[0]
            ) // conv.stride[0] + 1
            width = (width + 2 * conv.padding[1] - conv.kernel_size[1]) // conv.stride[
                1
            ] + 1
        return height, width

    def forward(self, data):
        """
        Forward pass through the network.
        """
        board, var = data

        # Process the board through CNN layers
        x = F.relu(self.bn1(self.conv1(board)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the CNN output
        x = x.view(x.size(0), -1)

        # Fully connected layers for board features
        x = F.relu(self.fc_img1(x))
        x = F.relu(self.fc_img2(x))

        # Process the variable input through FC layers
        y = F.relu(self.fc_var1(var))
        y = F.relu(self.fc_var2(y))

        # Combine image features and variable features
        combined = torch.cat((x, y), dim=1)

        # Fully connected layers for combined features
        combined = F.relu(self.fc_combined1(combined))
        combined = self.dropout_combined1(combined)  # Dropout applied here

        # Output layers
        pi = self.fc_combined2(combined)  # Action probabilities
        v = self.fc_value(combined)  # State value

        return F.log_softmax(pi, dim=1), torch.tanh(v)
