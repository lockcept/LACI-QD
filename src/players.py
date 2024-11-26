"""
Module players
"""

from functools import partial
import numpy as np

from board import Board
from gui import GUIQuoridor
from game import Game
from mcts import MCTS
from nnet_wrapper import NNetWrapper
from utils import Docdict


class Player:
    """
    Board player class.
    """

    def __init__(self, game: Game):
        self.game = game

    def play(self, board: Board, reverse_x: bool):
        """
        Play a move on the board.
        """


class RandomPlayer(Player):
    """
    Handle the random player actions.
    """

    def __init__(self, game: Game):

        Player.__init__(self, game)
        self.game = game

    def play(self, board: Board, reverse_x: bool):
        valids = self.game.get_valid_actions(board)
        valids = valids / np.sum(valids)
        return np.random.choice(len(valids), p=valids), valids


class HumanPlayer(Player):
    """
    Handle the human player actions.
    """

    def __init__(self, game, gui: GUIQuoridor):
        super().__init__(game)
        self.gui = gui

    def play(self, board, reverse_x=False):
        valid = self.game.get_valid_actions(board)
        self.gui.selected_position = None
        self.gui.is_human_turn = True

        while self.gui.selected_position is None:
            self.gui.root.update()

        action_type, x, y = self.gui.selected_position

        # 액션 생성
        if action_type == "move":
            if reverse_x:
                x = self.game.n - 1 - x
            action = x * self.game.n + y
        elif action_type == "h_wall":
            if reverse_x:
                x = self.game.n - 2 - x
            action = self.game.n * self.game.n + x * (self.game.n - 1) + y
        elif action_type == "v_wall":
            if reverse_x:
                x = self.game.n - 2 - x
            action = (
                self.game.n * self.game.n
                + x * (self.game.n - 1)
                + y
                + (self.game.n - 1) * (self.game.n - 1)
            )
        else:
            print("Invalid action type.")
            return self.play(board, reverse_x=reverse_x)

        # 유효성 검사
        if valid[action]:
            return action, None
        else:
            print("Invalid move, try again.")
            return self.play(board, reverse_x=reverse_x)


class MctsPlayer(Player):
    """
    Handle the MCTS player actions.
    """

    def __init__(self, game):
        Player.__init__(self, game)
        self.game = game
        nnet_wrapper = NNetWrapper(game)
        nnet_wrapper.load_checkpoint("./models", "best.pth.tar")
        self.mcts = MCTS(
            game=game,
            pi_v_function=nnet_wrapper.get_pi_v,
            args=Docdict({"numMCTSSims": 25, "cpuct": 1.0}),
        )

    def play(self, board, reverse_x):
        temp = 1
        pi = self.mcts.get_action_prob(board, temp=temp)
        return np.random.choice(len(pi), p=pi), pi


def greedy_function(game: Game, board: Board):
    """
    return the greedy action probabilities and the score
    """
    valids = game.get_valid_actions(board)
    scores = []

    for action, is_valid in enumerate(valids):
        if is_valid:
            next_board, _ = game.get_next_state(board, 1, action)

            my_distance = next_board.get_distance_to_goal(1)
            enemy_distance = next_board.get_distance_to_goal(-1)
            score = -(my_distance - enemy_distance) / game.n**2
            scores.append((action, score))

    max_score = max(scores, key=lambda x: x[1])[1]

    best_actions = [action for action, score in scores if score == max_score]

    action_probabilities = np.zeros_like(valids, dtype=float)
    probability = 1 / len(best_actions)
    for action in best_actions:
        action_probabilities[action] = probability

    return action_probabilities, max_score


class GreedyPlayer(Player):
    """
    Handle the greedy player actions.
    """

    def __init__(self, game: Game):
        super().__init__(game)
        self.game = game

    def play(self, board, reverse_x):
        action_probabilities, _ = greedy_function(self.game, board)

        best_action = np.random.choice(
            len(action_probabilities), p=action_probabilities
        )

        return best_action, action_probabilities


class GreedyMctsPlayer(Player):
    """
    Handle the greedy with MCTS player actions.
    """

    def __init__(self, game: Game):
        super().__init__(game)
        pi_v_function = partial(greedy_function, game)
        self.mcts = MCTS(
            game, pi_v_function, Docdict({"numMCTSSims": 25, "cpuct": 0.3})
        )

    def play(self, board, reverse_x):
        temp = 1
        pi = self.mcts.get_action_prob(board, temp=temp)
        return np.random.choice(len(pi), p=pi), pi
