import numpy as np

from board import Board
from gui import GUIQuoridor
from game import Game
from mcts import MCTS


class Player:
    def __init__(self, game: Game):
        self.game = game

    def play(self, board: Board, cur_player: int):
        pass


class RandomPlayer(Player):
    def __init__(self, game: Game):
        Player.__init__(self, game)
        self.game = game

    def play(self, board, cur_player):
        valids = self.game.get_valid_actions(board)
        valids = valids / np.sum(valids)
        return np.random.choice(len(valids), p=valids), valids


class HumanPlayer(Player):
    def __init__(self, game, gui: GUIQuoridor):
        super().__init__(game)
        self.gui = gui

    def play(self, board, cur_player):
        valid = self.game.get_valid_actions(board)
        self.gui.selected_position = None
        self.gui.is_human_turn = True

        print("Your turn! Select a position or place a wall.")

        while self.gui.selected_position is None:
            self.gui.root.update()

        action_type, x, y = self.gui.selected_position
        print(f"You selected: {action_type} at ({x}, {y})")

        # 액션 생성
        if action_type == "move":
            if cur_player == -1:
                x = self.game.n - 1 - x
            action = x * self.game.n + y
        elif action_type == "h_wall":
            if cur_player == -1:
                x = self.game.n - 2 - x
            action = self.game.n * self.game.n + x * (self.game.n - 1) + y
        elif action_type == "v_wall":
            if cur_player == -1:
                x = self.game.n - 2 - x
            action = (
                self.game.n * self.game.n
                + x * (self.game.n - 1)
                + y
                + (self.game.n - 1) * (self.game.n - 1)
            )
        else:
            print("Invalid action type.")
            return self.play(board, cur_player)

        # 유효성 검사
        if valid[action]:
            return action, None
        else:
            print("Invalid move, try again.")
            return self.play(board, cur_player)


class MCTSPlayer(Player):
    def __init__(self, game, mcts: MCTS):
        Player.__init__(self, game)
        self.game = game
        self.mcts = mcts

    def play(self, board, cur_player):
        temp = 1
        pi = self.mcts.getActionProb(board, temp=temp)
        return np.random.choice(len(pi), p=pi), pi
