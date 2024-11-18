import numpy as np

from Board import Board
from Game import Game
from MCTS import MCTS


class Player:
    def __init__(self, game: Game):
        self.game = game

    def play(self, board: Board):
        pass


class RandomPlayer(Player):
    def __init__(self, game):
        Player.__init__(self, game)
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        valid_moves = np.nonzero(valids)[0]
        return np.random.choice(valid_moves)


class HumanQuoridorPlayer(Player):
    def __init__(self, game):
        Player.__init__(self, game)
        self.game = game

    def play(self, board):
        # 사용자 입력 받기
        valid = self.game.getValidMoves(board, 1)
        print(valid)
        self.game.display(board)
        while True:
            a = input("Enter your move: ")
            x, y, w = map(int, a.split())
            action = None
            if w == 0:
                action = x * self.game.n + y
            else:
                action = (
                    self.game.n * self.game.n
                    + x * (self.game.n - 1)
                    + y
                    + (self.game.n - 1) * (self.game.n - 1) * (w - 1)
                )
            if valid[action]:
                break
            else:
                print("Invalid move")
        return action


class MCTSPlayer(Player):
    def __init__(self, game, mcts: MCTS):
        Player.__init__(self, game)
        self.game = game
        self.mcts = mcts

    def play(self, board):
        # MCTS를 사용하여 최적의 움직임을 선택
        temp = 1
        pi = self.mcts.getActionProb(board, temp=temp)
        return np.random.choice(len(pi), p=pi)
