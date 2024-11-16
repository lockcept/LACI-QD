import logging
import coloredlogs
from Coach import Coach
from Game import Game
from Trainer import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level="INFO")

args = dotdict(
    {
        "gameSize": 5,
        "numIters": 100,
        "numEps": 1,
        "tempThreshold": 15,  #
        "updateThreshold": 0.6,
        "maxlenOfQueue": 200000,
        "numMCTSSims": 25,
        "arenaCompare": 40,
        "cpuct": 1,
        "checkpoint": "./temp/",
        "load_model": False,
        "load_folder_file": ("./temp", "best.pth.tar"),
        "numItersForTrainExamplesHistory": 20,
    }
)


def main():
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

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info("Starting the learning process 🎉")
    c.learn()


if __name__ == "__main__":
    main()
