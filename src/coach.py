"""
Coach module
"""

from functools import partial
import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from typing import Any
import torch.multiprocessing as mp

import numpy as np
from tqdm import tqdm

from arena import Arena
from mcts import MCTS
from game import Game
from nnet_wrapper import NNetWrapper

log = logging.getLogger(__name__)


def action_function(x, mcts):
    """
    return the action to take given the board state x
    """
    return np.argmax(mcts.get_action_prob(x, temp=0))


def execute_episode_worker(args: tuple[Game, NNetWrapper, Any]):
    """
    Worker function for multiprocessing.
    Each worker gets its own MCTS instance to run an episode.
    """
    game, nnet_wrapper, coach_args = args
    mcts = MCTS(game, nnet_wrapper.get_pi_v, coach_args)
    train_examples = []
    board = game.get_init_board()
    player = 1
    episode_step = 0

    while True:
        episode_step += 1
        canonical_board = board.get_canonical_form(player)
        temp = int(episode_step < coach_args.tempThreshold)

        pi = mcts.get_action_prob(canonical_board, temp=temp)
        sym = game.get_symmetries(canonical_board, pi)
        for b, p in sym:
            train_examples.append([game.board_to_input(b), player, p, None])

        action = np.random.choice(len(pi), p=pi)
        board, player = game.get_next_state(board, player, action)

        r = game.get_win_status(board, player)

        if r is not None:
            return [
                (x[0], x[2], r * ((-1) ** (x[1] != player))) for x in train_examples
            ]


class Coach:
    """
    Self play and save examples for training, and train the neural network.
    """

    def __init__(self, game: Game, nnet_wrapper: NNetWrapper, args):
        self.game = game
        self.nnet_wrapper = nnet_wrapper
        self.pnet_wrapper = self.nnet_wrapper.__class__(
            self.game
        )  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet_wrapper.get_pi_v, self.args)
        self.train_examples_history = (
            []
        )  # history of examples from args.numItersFortrain_examples_history latest iterations

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        for i in range(1, self.args.numIters + 1):
            log.info(f"Starting Iter #{i} ...")

            if i == 1 and self.train_examples_history:
                print("Skipping self-play for the first iteration")
            else:
                # examples of the iteration
                iteration_train_examples = self.run_self_play()

                # save the iteration examples to the history
                self.train_examples_history.append(iteration_train_examples)

            if (
                len(self.train_examples_history)
                > self.args.numItersForTrainExamplesHistory
            ):
                log.warning(
                    "Removing the oldest entry in trainExamples. len(train_examples_history) = %d",
                    len(self.train_examples_history),
                )
                self.train_examples_history.pop(0)

            # backup history to a file
            self.save_train_examples()

            # shuffle examples before training
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.nnet_wrapper.save_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            self.pnet_wrapper.load_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            pmcts = MCTS(self.game, self.pnet_wrapper.get_pi_v, self.args)

            self.nnet_wrapper.train(train_examples, num_iter=i)
            nmcts = MCTS(self.game, self.nnet_wrapper.get_pi_v, self.args)

            log.info("PITTING AGAINST PREVIOUS VERSION")
            arena = Arena(
                partial(action_function, mcts=pmcts),
                partial(action_function, mcts=nmcts),
                self.game,
            )
            pwins, nwins, draws = arena.play_games(
                self.args.arenaCompare, num_iter=i, num_processes=self.args.numProcesses
            )

            log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d", nwins, pwins, draws)
            if (
                pwins + nwins == 0
                or float(nwins) / (pwins + nwins) < self.args.updateThreshold
            ):
                log.info("REJECTING NEW MODEL")
                self.nnet_wrapper.load_checkpoint(
                    folder=self.args.checkpoint, filename="temp.pth.tar"
                )
            else:
                log.info("ACCEPTING NEW MODEL")
                self.nnet_wrapper.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.pth.tar"
                )

            self.nnet_wrapper.save_checkpoint(
                folder=self.args.checkpoint,
                filename=self.get_checkpoint_file_name(i),
            )

    def run_self_play(self):
        """
        Runs self-play episodes
        """
        log.info(f"Starting {self.args.numEps} episodes of self-play")
        process_args = [
            (self.game, self.nnet_wrapper, self.args) for _ in range(self.args.numEps)
        ]

        if self.args.numProcesses == 1:
            results = []
            for args in tqdm(process_args, desc="Self Play"):
                results.append(execute_episode_worker(args))
        else:
            with mp.Pool(processes=self.args.numProcesses) as pool:
                results = list(
                    tqdm(
                        pool.imap(execute_episode_worker, process_args),
                        total=len(process_args),
                        desc="Self Play",
                    )
                )

        # Collect and return training examples from all episodes
        iteration_train_examples = deque([], maxlen=self.args.maxlenOfQueue)
        for episode_examples in results:
            iteration_train_examples += episode_examples

        return iteration_train_examples

    def get_checkpoint_file_name(self, iteration):
        """
        Get the name of the checkpoint file
        """
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def save_train_examples(self):
        """
        Saves the trainExamples to a file
        """
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, "last_checkpoint.exmaples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)
        f.close()

    def load_train_examples(self):
        """
        Load the trainExamples from file
        """
        examples_file = os.path.join(
            self.args.load_examples_file[0], self.args.load_examples_file[1]
        )
        if not os.path.isfile(examples_file):
            log.warning(f'File "{examples_file}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examples_file, "rb") as f:
                self.train_examples_history = Unpickler(f).load()
            log.info("Loading done!")
