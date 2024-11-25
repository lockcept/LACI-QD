import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from multiprocessing import Pool, cpu_count
from typing import Any

import numpy as np
from tqdm import tqdm

from arena import Arena
from mcts import MCTS
from game import Game
from nnet_wrapper import NNetWrapper

log = logging.getLogger(__name__)


def execute_episode_worker(args: tuple[Game, NNetWrapper, Any]):
    """
    Worker function for multiprocessing.
    Each worker gets its own MCTS instance to run an episode.
    """
    game, nnet, coach_args = args
    mcts = MCTS(game, nnet, coach_args)
    trainExamples = []
    board = game.get_init_board()
    curPlayer = 1
    episodeStep = 0

    while True:
        episodeStep += 1
        canonicalBoard = board.get_canonical_form(curPlayer)
        temp = int(episodeStep < coach_args.tempThreshold)

        pi = mcts.getActionProb(canonicalBoard, temp=temp)
        sym = game.get_symmetries(canonicalBoard, pi)
        for b, p in sym:
            trainExamples.append([game.board_to_input(b), curPlayer, p, None])

        action = np.random.choice(len(pi), p=pi)
        board, curPlayer = game.get_next_state(board, curPlayer, action)

        r = game.get_win_status(board, curPlayer)

        if r is not None:
            return [
                (x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples
            ]


class Coach:
    def __init__(self, game: Game, nnet: NNetWrapper, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = (
            []
        )  # history of examples from args.numItersForTrainExamplesHistory latest iterations

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f"Starting Iter #{i} ...")

            # examples of the iteration
            iterationTrainExamples = self.run_self_play_parallel()

            # save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

            if (
                len(self.trainExamplesHistory)
                > self.args.numItersForTrainExamplesHistory
            ):
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}"
                )
                self.trainExamplesHistory.pop(0)

            # backup history to a file
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            self.pnet.load_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info("PITTING AGAINST PREVIOUS VERSION")
            arena = Arena(
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game,
            )
            pwins, nwins, draws = arena.play_games(self.args.arenaCompare, num_iter=i)

            log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
            if (
                pwins + nwins == 0
                or float(nwins) / (pwins + nwins) < self.args.updateThreshold
            ):
                log.info("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(
                    folder=self.args.checkpoint, filename="temp.pth.tar"
                )
            else:
                log.info("ACCEPTING NEW MODEL")
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=self.getCheckpointFile(i)
                )
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.pth.tar"
                )

    def run_self_play_parallel(self):
        """
        Runs self-play episodes in parallel using multiprocessing.
        """
        log.info(f"Starting {self.args.numEps} episodes of self-play in parallel.")
        process_args = [
            (self.game, self.nnet, self.args) for _ in range(self.args.numEps)
        ]

        with Pool(processes=cpu_count() - 3) as pool:
            results = list(
                tqdm(
                    pool.imap(execute_episode_worker, process_args),
                    total=self.args.numEps,
                    desc="Self Play",
                )
            )

        # Collect and return training examples from all episodes
        iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        for episode_examples in results:
            iterationTrainExamples += episode_examples

        return iterationTrainExamples

    def getCheckpointFile(self, iteration):
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(
            self.args.load_folder_file[0], self.args.load_folder_file[1]
        )
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info("Loading done!")
