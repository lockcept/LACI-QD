"""
Module players
"""

import numpy as np

from board import Board
from gui import GUIQuoridor
from game import Game
from mcts import MCTS


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


class MCTSPlayer(Player):
    """
    Handle the MCTS player actions.
    """

    def __init__(self, game, mcts: MCTS):
        Player.__init__(self, game)
        self.game = game
        self.mcts = mcts

    def play(self, board, reverse_x):
        temp = 1
        pi = self.mcts.get_action_prob(board, temp=temp)
        return np.random.choice(len(pi), p=pi), pi


class GreedyPlayer(Player):
    """
    Handle the greedy player actions.
    """

    def __init__(self, game: Game):
        super().__init__(game)
        self.game = game

    def play(self, board, reverse_x):
        valids = self.game.get_valid_actions(board)
        distances = []

        for action, is_valid in enumerate(valids):
            if is_valid:
                next_board, _ = self.game.get_next_state(board, 1, action)

                my_distance = next_board.get_distance_to_goal(1)
                enemy_distance = next_board.get_distance_to_goal(-1)
                distances.append((action, my_distance - enemy_distance))

        min_distance = min(distances, key=lambda x: x[1])[1]

        best_actions = [
            action for action, distance in distances if distance == min_distance
        ]

        action_probabilities = np.zeros_like(valids, dtype=float)
        probability = 1 / len(best_actions)
        for action in best_actions:
            action_probabilities[action] = probability

        best_action = np.random.choice(best_actions)

        return best_action, action_probabilities
