import numpy as np

from Board import Board
from GUI import GUIQuoridor
from Game import Game
from MCTS import MCTS


class Player:
    def __init__(self, game: Game):
        self.game = game

    def play(self, board: Board):
        pass


class RandomPlayer(Player):
    def __init__(self, game: Game):
        Player.__init__(self, game)
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board)
        valid_moves = np.nonzero(valids)[0]
        return np.random.choice(valid_moves)


class HumanPlayer(Player):
    def __init__(self, game, gui: GUIQuoridor):
        super().__init__(game)
        self.gui = gui

    def play(self, board):
        """
        GUI를 통해 사용자의 액션을 대기하고 반환합니다.
        """
        valid = self.game.getValidMoves(board)
        self.gui.selected_position = None

        print("Your turn! Select a position or place a wall.")

        while self.gui.selected_position is None:
            self.gui.root.update()  # GUI 이벤트 루프 실행

        action_type, x, y = self.gui.selected_position
        print(f"You selected: {action_type} at ({x}, {y})")

        # 액션 생성
        if action_type == "move":
            action = x * self.game.n + y
        elif action_type == "h_wall":
            action = self.game.n * self.game.n + x * (self.game.n - 1) + y
        elif action_type == "v_wall":
            action = (
                self.game.n * self.game.n
                + x * (self.game.n - 1)
                + y
                + (self.game.n - 1) * (self.game.n - 1)
            )
        else:
            print("Invalid action type.")
            return self.play(board)  # 잘못된 입력이면 다시 요청

        # 유효성 검사
        if valid[action]:
            return action
        else:
            print("Invalid move, try again.")
            return self.play(board)


class MCTSPlayer(Player):
    def __init__(self, game, mcts: MCTS):
        Player.__init__(self, game)
        self.game = game
        self.mcts = mcts

    def play(self, board):
        temp = 1
        pi = self.mcts.getActionProb(board, temp=temp)
        return np.random.choice(len(pi), p=pi)
