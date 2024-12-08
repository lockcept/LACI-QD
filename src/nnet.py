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
        self.board_x, self.board_y, self.board_z = board_size  # 5 x 17 x 17 for n = 9
        self.var_size = var_size
        self.action_size = game.get_action_size()
        self.args = args

        out_channels = [64, 128, 256, 512]

        # CNN for image input
        self.conv1 = nn.Conv2d(
            in_channels=self.board_x,
            out_channels=out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )  # 17x17 -> 17x17

        self.conv2 = nn.Conv2d(
            in_channels=out_channels[0],
            out_channels=out_channels[1],
            kernel_size=5,
            stride=2,
            padding=2,
        )  # 17x17 -> 9x9

        self.conv3 = nn.Conv2d(
            in_channels=out_channels[1],
            out_channels=out_channels[2],
            kernel_size=3,
            stride=1,
            padding=1,
        )  # 9x9 -> 9x9

        self.conv4 = nn.Conv2d(
            in_channels=out_channels[2],
            out_channels=out_channels[3],
            kernel_size=3,
            stride=1,
            padding=1,
        )  # 9x9 -> 9x9

        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.bn4 = nn.BatchNorm2d(out_channels[3])

        # Calculate dynamic output size after convolutions
        final_board_x, final_board_y = self.calculate_conv_output(
            self.board_y, self.board_z
        )

        full_channels = [512, 256]

        # Fully connected layers for image features
        self.fc_img1 = nn.Linear(
            out_channels[3] * final_board_x * final_board_y, full_channels[0]
        )
        self.fc_img2 = nn.Linear(full_channels[0], full_channels[1])

        # Fully connected layers for variables
        self.fc_var1 = nn.Linear(var_size, 32)
        self.fc_var2 = nn.Linear(32, 32)

        # Combined fully connected layers
        self.fc_combined1 = nn.Linear(full_channels[1] + 32, 256)
        self.dropout_combined1 = nn.Dropout(args.dropout)  # Dropout added here
        self.fc_combined2 = nn.Linear(256, self.action_size)
        self.fc_value = nn.Linear(256, 1)

    def calculate_conv_output(self, height, width):
        """
        Dynamically calculates the output size after Conv2d layers.
        """
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
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
        x = F.relu(self.bn4(self.conv4(x)))

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
