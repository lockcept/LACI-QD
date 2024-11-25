"""
Game is a wrapper for the Board class with metadata.
"""

import numpy as np
from board import Board


class Game:
    """
    Game class to handle the Board class and game metadata.
    """

    def __init__(self, n: int):
        self.n = n
        self.max_turn = n * n * 2
        self.winning_criteria = 0.8

    def get_init_board(self):
        """
        Initializes and returns the initial game board.
        """
        return Board(self.n)

    def get_board_size(self):
        """
        Returns the size of the game board to learn network.

        Returns:
            tuple: A tuple containing the dimensions of the board
        """
        return (
            self.n,
            self.n,
            6,
        )  # p1_pos, p2_pos, h_walls, v_walls, p1_walls, p2_walls

    def get_action_size(self):
        """
        Calculate the total number of possible actions in the game.

        Returns:
            int: The total number of possible moves and walls.
        """

        return self.n * self.n + (self.n - 1) * (self.n - 1) * 2

    def get_next_state(self, board: Board, player: int, action: int):
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

    def get_valid_actions(self, board: Board):
        """
        board: current board
        return: a binary vector of length self.get_action_size(), 1 for all valid moves, 0 for others
        """
        valids = [0] * self.get_action_size()

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

    def get_win_status(self, board: Board, player):
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

        if player == 1:
            if board.p1_pos[0] == self.n - 1:
                return 1
            if board.p2_pos[0] == 0:
                return -1
        else:  # player == -1
            if board.p2_pos[0] == 0:
                return 1
            if board.p1_pos[0] == self.n - 1:
                return -1

        return None  # Game has not ended

    def get_canonical_pi(self, pi, player) -> list[float]:
        """
        Transforms the policy vector `pi` to the canonical form for the given player.
        Args:
            pi (list[float]): The policy vector to be transformed.
            player (int): The player for whom the canonical form is to be generated.
                          If player is -1, the policy vector is transformed.

        Returns:
            list[float]: The canonical form of the policy vector.
        """
        if player == -1:
            n = self.n
            pi_board = np.reshape(pi[: n * n], (n, n))
            pi_walls = np.reshape(pi[n * n :], (2, n - 1, n - 1))

            pi_board = np.flip(pi_board, axis=0)
            pi_walls = np.flip(pi_walls, axis=1)

            pi = list(pi_board.ravel()) + list(pi_walls.ravel())

        return pi

    def get_symmetries(self, board: Board, pi) -> list[tuple[Board, list[float]]]:
        """
        Generate symmetrical versions of the board and policy vector.
        Args:
            board (Board): The current state of the board.
            pi (list[float]): The policy vector, which includes action probabilities.

        Returns:
            list[tuple[Board, list[float]]]: A list of tuples, where each tuple contains
            a symmetrical version of the board and the corresponding policy vector.
        """
        assert len(pi) == self.get_action_size()

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
