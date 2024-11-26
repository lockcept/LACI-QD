"""
This file contains the NNetWrapper class, which is a wrapper around the NNet class.
"""

import csv
import os

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from board import Board
from game import Game
from utils import Docdict, AverageMeter


from nnet import NNet

args = Docdict(
    {
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 10,
        "batch_size": 64,
        "cuda": torch.cuda.is_available(),
        "num_channels": 512,
    }
)

CSV_FILE_PATH = "logs/train.csv"


class NNetWrapper:
    """
    This class is a wrapper around the NNet class.
    """

    def __init__(self, game: Game):
        self.game = game
        self.nnet = NNet(game, args)
        board_size, var_size = game.get_input_size()
        self.board_x, self.board_y, self.board_z = board_size
        self.var_size = var_size

        if args.cuda:
            self.nnet.cuda()

        with open(CSV_FILE_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["num_iter", "num_epoch", "pi_loss", "v_loss"])

    def train(self, examples, num_iter):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc="Training Net")
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                data, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                boards, var_lists = zip(*data)

                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                var_lists = torch.FloatTensor(np.array(var_lists).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # Move to GPU if CUDA is enabled
                if args.cuda:
                    boards, var_lists, target_pis, target_vs = (
                        boards.contiguous().cuda(),
                        var_lists.contiguous().cuda(),
                        target_pis.contiguous().cuda(),
                        target_vs.contiguous().cuda(),
                    )

                # Compute output
                out_pi, out_v = self.nnet(
                    (boards, var_lists)
                )  # Pass both boards and var_lists
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # Record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # Compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            with open(CSV_FILE_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([num_iter, epoch + 1, pi_losses, v_losses])

    def predict(self, data):
        """
        data: (board, var_list)
        """
        board, var_list = data
        board = torch.FloatTensor(board.astype(np.float64))
        var = torch.FloatTensor(var_list.astype(np.float64))

        if args.cuda:
            board = board.contiguous().cuda()
            var = var.contiguous().cuda()

        board = board.view(1, self.board_x, self.board_y, self.board_z)
        var = var.view(1, self.var_size)

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet((board, var))

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        """
        Compute the loss for the policy network.
        """
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        """
        Compute the loss for the value network.
        """
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        """
        Save the current model to a checkpoint.
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint Directory does not exist! Making directory {folder}")
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save(
            {
                "state_dict": self.nnet.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        """
        Load the model from a checkpoint.
        """
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        map_location = None if args.cuda else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
        self.nnet.load_state_dict(checkpoint["state_dict"])

    def get_pi_v(self, board: Board):
        """
        Get the policy and value of the board.
        """
        return self.predict(self.game.board_to_input(board))
