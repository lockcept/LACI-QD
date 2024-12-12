"""
This script sets up and starts the training process for a Quoridor Game AI using a neural network.

Usage:
    Run this script directly to start the training process.
"""

import logging
import os
import coloredlogs
import torch.multiprocessing as mp
from coach import Coach
from game import Game
from nnet_wrapper import NNetWrapper as nn
from utils import Docdict

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")

args = Docdict(
    {
        "gameSize": 9,
        "numIters": 100,
        "numEps": 50,
        "tempThreshold": 15,  #
        "updateThreshold": 0.6,
        "maxlenOfQueue": 200000,
        "numMCTSSims": 50,
        "arenaCompare": 20,
        "cpuct": 1,
        "checkpoint": "./models/",
        "load_model": False,
        "load_folder_file": ("./models", "best.pth.tar"),
        "load_examples": True,
        "load_examples_file": ("./examples", "prepared.examples"),
        "numItersForTrainExamplesHistory": 20,
        "numProcesses": 1,
    }
)


def main():
    """
    Main function to initialize and start the learning process.
    """

    log_path = "logs/"
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log.info("Loading %s...", Game.__name__)
    g = Game(args.gameSize)

    log.info("Loading %s...", nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info(
            'Loading checkpoint "%s/%s"...',
            args.load_folder_file[0],
            args.load_folder_file[1],
        )
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning("Not loading a checkpoint!")

    log.info("Loading the Coach...")
    c = Coach(g, nnet, args)

    if args.load_examples:
        log.info("Loading 'trainExamples' from file...")
        c.load_train_examples()

    log.info("Starting the learning process 🎉")
    c.learn()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
