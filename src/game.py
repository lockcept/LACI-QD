"""
A class to represent a game. It is just a wrapper for the Board class with metadata about the game.
"""

import numpy as np
from board import Board


class Game:
    def __init__(self, n: int):
        self.n = n
        self.max_turn = n * n * 2
        self.winning_criteria = 0.8

    def get_init_board(self):
        """
        Initializes and returns the initial game board.
        """
        return Board(self.n)

    def getBoardSize(self):
        return (
            self.n,
            self.n,
            6,
        )  # p1_pos, p2_pos, h_walls, v_walls, p1_walls, p2_walls

    def getActionSize(self):
        """
        all possible moves + all possible walls
        """
        return self.n * self.n + (self.n - 1) * (self.n - 1) * 2

    def getNextState(self, board: Board, player: int, action: int):
        """
        board: current board
        player: current player (1 or -1)
        action: action taken by current player (assume it is legal)
        return: next board and player
        """

        board = board.get_canonical_form(player)

        if action < self.n * self.n:
            move = (action // self.n, action % self.n)
            board.execute_move(move)
        else:
            wall_type = (action - self.n * self.n) // ((self.n - 1) * (self.n - 1))
            wall_pos = (action - self.n * self.n) % ((self.n - 1) * (self.n - 1))
            wall_pos = (wall_pos // (self.n - 1), wall_pos % (self.n - 1))
            board.place_wall(wall_pos, wall_type)

        original_board = board.get_canonical_form(player)

        return (original_board, -player)

    def getValidMoves(self, board: Board):
        """
        board: current board
        return: a binary vector of length self.getActionSize(), 1 for all valid moves, 0 for others
        """
        valids = [0] * self.getActionSize()

        # Check if move is valid
        moves = board.get_legal_moves()
        for move in moves:
            valids[move[0] * self.n + move[1]] = 1

        # Check if wall is valid
        walls = board.get_legal_walls()
        for wall in walls:
            wall_idx = self.n * self.n + wall[0] * (self.n - 1) + wall[1]
            if wall[2] == "h":
                wall_idx += 0
            else:
                wall_idx += (self.n - 1) * (self.n - 1)
            valids[wall_idx] = 1

        return np.array(valids)

    def getGameEnded(self, board: Board, player):
        """
        board: current board
        player: current player (1 or -1)
        return: None if game has not ended. return score if game has ended.
        """
        if board.turns == self.max_turn:
            p1_dist = self.n - 1 - board.p1_pos[0]
            p2_dist = board.p2_pos[0]
            p1_win_ratio = (p2_dist - p1_dist) / self.n
            if player == 1:
                return p1_win_ratio * self.winning_criteria
            else:
                return -(p1_win_ratio * self.winning_criteria)

        if board.is_win(player):
            return 1
        if board.is_win(-player):
            return -1
        return None

    def get_canonical_pi(self, pi, player) -> list[float]:
        if player == -1:
            n = self.n
            pi_board = np.reshape(pi[: n * n], (n, n))
            pi_walls = np.reshape(pi[n * n :], (2, n - 1, n - 1))

            pi_board = np.flip(pi_board, axis=0)
            pi_walls = np.flip(pi_walls, axis=1)

            pi = list(pi_board.ravel()) + list(pi_walls.ravel())

        return pi

    def getSymmetries(self, board: Board, pi) -> list[tuple[Board, list[float]]]:
        assert len(pi) == self.getActionSize()

        n = self.n
        pi_board = np.reshape(pi[: n * n], (n, n))
        pi_walls = np.reshape(pi[n * n :], (2, n - 1, n - 1))

        symmetries = []

        # 원래 모습
        symmetries.append((board, pi))

        # 좌우 대칭
        board_lr = board.get_flipped_form()
        pi_board_lr = np.fliplr(pi_board)
        pi_walls_lr = np.flip(pi_walls, axis=2)
        pi_lr = list(pi_board_lr.ravel()) + list(pi_walls_lr.ravel())
        symmetries.append((board_lr, pi_lr))

        return symmetries

    @staticmethod
    def display(board: Board):
        n = board.n
        board_size_with_wall = 2 * n - 1
        display_board = np.full(
            (board_size_with_wall, board_size_with_wall), " ", dtype=str
        )

        for x in range(n):
            for y in range(n):
                display_board[x * 2, y * 2] = "□"

        display_board[board.p1_pos[0] * 2, board.p1_pos[1] * 2] = "●"
        display_board[board.p2_pos[0] * 2, board.p2_pos[1] * 2] = "■"

        for x, y in board.h_walls:
            display_board[x * 2 + 1, y * 2] = "━"
            display_board[x * 2 + 1, y * 2 + 1] = "━"
            display_board[x * 2 + 1, y * 2 + 2] = "━"
        for x, y in board.v_walls:
            display_board[x * 2, y * 2 + 1] = "┃"
            display_board[x * 2 + 1, y * 2 + 1] = "┃"
            display_board[x * 2 + 2, y * 2 + 1] = "┃"

        print("  ", end="")
        for y in range(board_size_with_wall):
            print(y % 10, end=" ")
        print("")

        for x in range(board_size_with_wall):
            print(x % 10, end=" ")
            for y in range(board_size_with_wall):
                print(display_board[x, y], end=" ")
            print("")
        print(
            "wall 1: ",
            board.p1_walls,
            ", ",
            "wall 2: ",
            board.p2_walls,
        )
