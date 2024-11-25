"""
This module defines the Board class for the game Quoridor.

Classes:
    Board: Represents the game board and handles game logic.
"""

from collections import deque
import numpy as np


class Board:
    """
    A class to represent the game board for a two-player game.
    """

    def __init__(self, n):
        self.n = n
        self.p1_pos = (0, n // 2)
        self.p2_pos = (n - 1, n // 2)
        self.p1_walls = n * n // 8
        self.p2_walls = n * n // 8
        self.h_walls = set()
        self.v_walls = set()
        self.turns = 0

    def get_canonical_form(self, player):
        """
        Returns the canonical form of the board for the given player.

        Args:
            player (int): The player for whom the canonical form is to be generated.
                          Should be either 1 or -1.

        Returns:
            Board: A new Board object representing the canonical form of the board.
        """
        if player == 1:
            board = Board(self.n)
            board.p1_pos = self.p1_pos
            board.p2_pos = self.p2_pos
            board.p1_walls = self.p1_walls
            board.p2_walls = self.p2_walls
            board.h_walls = set([(x, y) for x, y in self.h_walls])
            board.v_walls = set([(x, y) for x, y in self.v_walls])
            board.turns = self.turns
            return board
        else:
            board = Board(self.n)
            board.p1_pos = (self.n - 1 - self.p2_pos[0], self.p2_pos[1])
            board.p2_pos = (self.n - 1 - self.p1_pos[0], self.p1_pos[1])
            board.p1_walls = self.p2_walls
            board.p2_walls = self.p1_walls
            board.h_walls = set([(self.n - 2 - x, y) for x, y in self.h_walls])
            board.v_walls = set([(self.n - 2 - x, y) for x, y in self.v_walls])
            board.turns = self.turns
            return board

    def get_flipped_form(self):
        """
        Returns a new Board object with the positions and walls flipped vertically.

        Returns:
            Board: A new Board instance with flipped positions and walls.
        """
        board = Board(self.n)
        board.p1_pos = (self.p1_pos[0], self.n - 1 - self.p1_pos[1])
        board.p2_pos = (self.p2_pos[0], self.n - 1 - self.p2_pos[1])
        board.p1_walls = self.p1_walls
        board.p2_walls = self.p2_walls
        board.h_walls = set([(x, self.n - 2 - y) for x, y in self.h_walls])
        board.v_walls = set([(x, self.n - 2 - y) for x, y in self.v_walls])
        board.turns = self.turns
        return board

    def string_representation(self):
        """
        Generates a string representation of the board state.

        Returns:
            str: A string representing the current state of the board
        """

        def format_sorted_tuple_set(tuple_set):
            sorted_tuples = sorted(tuple_set, key=lambda x: (x[0], x[1]))
            return ",".join(f"({a},{b})" for a, b in sorted_tuples)

        return (
            f"({self.p1_pos[0]},{self.p1_pos[1]})"
            + ":"
            + f"({self.p2_pos[0]},{self.p2_pos[1]})"
            + ":"
            + format_sorted_tuple_set(self.h_walls)
            + ":"
            + format_sorted_tuple_set(self.v_walls)
            + ":"
            + str(self.p1_walls)
            + ":"
            + str(self.p2_walls)
            + ":"
            + str(self.turns)
        )

    def execute_move(self, move):
        """
        Executes a move for the player 1.

        Parameters:
        move (tuple): The new position to move to.

        Returns:
        None
        """
        self.p1_pos = move
        self.turns += 1

    def place_wall(self, pos, wall_type):
        """
        Places a wall on the board at the specified position for the player 1.

        Args:
            pos (tuple): The position where the wall is to be placed.
            wall_type (int): The type of wall to place (0 for horizontal, 1 for vertical).
        """
        if wall_type == 0:
            self.h_walls.add(pos)
        else:
            self.v_walls.add(pos)

        self.p1_walls -= 1
        self.turns += 1

    def is_win(self, player):
        if player == 1:
            return self.p1_pos[0] == self.n - 1
        else:
            return self.p2_pos[0] == 0

    def is_wall_between(self, pos1, pos2):
        """
        Determines if there is a wall between two adjacent positions on the board.

        Args:
            pos1 (tuple): The first position as a tuple (x1, y1).
            pos2 (tuple): The second position as a tuple (x2, y2).

        Returns:
            bool: True if there is a wall between the two positions, False otherwise.
        """
        x1, y1 = pos1
        x2, y2 = pos2

        if x1 == x2:
            y = min(y1, y2)
            if (x1 - 1, y) in self.v_walls or (x1, y) in self.v_walls:
                return True
        elif y1 == y2:
            x = min(x1, x2)
            if (x, y1 - 1) in self.h_walls or (x, y1) in self.h_walls:
                return True
        return False

    def get_legal_moves(self):
        """
        Returns a list of legal moves for player 1
        """
        pos = self.p1_pos
        opponent_pos = self.p2_pos

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        legal_moves = []

        for move in directions:
            new_pos = (pos[0] + move[0], pos[1] + move[1])
            if not self.is_wall_between(pos, new_pos):
                # 도착 위치에 적이 있는 경우
                if new_pos == opponent_pos:
                    jump_pos = (new_pos[0] + move[0], new_pos[1] + move[1])
                    if (
                        self.is_wall_between(new_pos, jump_pos)
                        or jump_pos[0] < 0
                        or jump_pos[0] >= self.n
                        or jump_pos[1] < 0
                        or jump_pos[1] >= self.n
                    ):
                        # 수직 방향 이동 검사
                        vertical_moves = []
                        if move[0] == 0:
                            vertical_moves = [(-1, 0), (1, 0)]
                        else:
                            vertical_moves = [(0, -1), (0, 1)]

                        for v_move in vertical_moves:
                            new_v_pos = (new_pos[0] + v_move[0], new_pos[1] + v_move[1])
                            if not self.is_wall_between(new_pos, new_v_pos):
                                legal_moves.append(new_v_pos)
                    else:
                        legal_moves.append(jump_pos)
                else:
                    legal_moves.append((pos[0] + move[0], pos[1] + move[1]))

        return [
            (x, y)
            for x, y in legal_moves
            if x >= 0 and x < self.n and y >= 0 and y < self.n
        ]

    def is_legal_wall(self, pos, wall_type):
        """
        Determines if placing a wall at the given position is legal.

        Args:
            pos (tuple): The position (row, column) where the wall is to be placed.
            wall_type (int): The type of wall to be placed. 0 for horizontal wall, 1 for vertical wall.

        Returns:
            bool: True if the wall placement is legal, False otherwise.

        The function checks the following conditions:
        1. The position is within the bounds of the board.
        2. The wall does not overlap with existing walls of the same type or intersect with walls of the opposite type.
        3. Both players can still reach their respective goals after placing the wall.
        """
        if pos[0] < 0 or pos[0] >= self.n - 1 or pos[1] < 0 or pos[1] >= self.n - 1:
            return False

        if wall_type == 0:  # 수평 벽
            if (
                pos in self.h_walls
                or (pos[0], pos[1] - 1) in self.h_walls
                or (pos[0], pos[1] + 1) in self.h_walls
                or (pos in self.v_walls)
            ):
                return False
        elif wall_type == 1:  # 수직 벽
            if (
                pos in self.v_walls
                or (pos[0] - 1, pos[1]) in self.v_walls
                or (pos[0] + 1, pos[1]) in self.v_walls
                or (pos in self.h_walls)
            ):
                return False

        temp_h_walls = self.h_walls.copy()
        temp_v_walls = self.v_walls.copy()
        if wall_type == 0:
            temp_h_walls.add(pos)
        else:
            temp_v_walls.add(pos)

        if not (
            self.can_reach_goal(self.p1_pos, 1, temp_h_walls, temp_v_walls)
            and self.can_reach_goal(self.p2_pos, 2, temp_h_walls, temp_v_walls)
        ):
            return False

        return True

    def can_reach_goal(self, start, player, h_walls, v_walls):
        """
        Determines if a player can reach their goal row on the board.

        Args:
            start (tuple): The starting position (x, y) of the player.
            player (int): The player number (1 or 2). Player 1 aims for the last row, player 2 aims for the first row.
            h_walls (set): A set of tuples representing the positions of horizontal walls.
            v_walls (set): A set of tuples representing the positions of vertical walls.

        Returns:
            bool: True if the player can reach their goal row, False otherwise.
        """
        goal_row = self.n - 1 if player == 1 else 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = deque([start])
        visited = set()
        visited.add(start)

        while queue:
            x, y = queue.popleft()
            if x == goal_row:
                return True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.n and 0 <= ny < self.n and (nx, ny) not in visited:
                    if (
                        (
                            dx == -1
                            and (x - 1, y - 1) not in h_walls
                            and (x - 1, y) not in h_walls
                        )
                        or (
                            dx == 1
                            and (x, y - 1) not in h_walls
                            and (x, y) not in h_walls
                        )
                        or (
                            dy == -1
                            and (x - 1, y - 1) not in v_walls
                            and (x, y - 1) not in v_walls
                        )
                        or (
                            dy == 1
                            and (x - 1, y) not in v_walls
                            and (x, y) not in v_walls
                        )
                    ):
                        queue.append((nx, ny))
                        visited.add((nx, ny))

        return False

    def get_legal_walls(self):
        """
        Returns a list of legal walls for player 1
        """
        legal_walls = []
        if self.p1_walls <= 0:
            return legal_walls
        for i in range(self.n - 1):
            for j in range(self.n - 1):
                if self.is_legal_wall((i, j), 0):
                    legal_walls.append((i, j, "h"))
                if self.is_legal_wall((i, j), 1):
                    legal_walls.append((i, j, "v"))
        return legal_walls

    def to_array(self):
        """
        Converts the board state to a 3D numpy array representation.

        Returns:
            np.ndarray: A 3D numpy array representing the board state.
        """
        board_array = np.zeros((self.n, self.n, 6))
        board_array[self.p1_pos][0] = 1
        board_array[self.p2_pos][1] = 1
        for wall in self.h_walls:
            board_array[wall][2] = 1
        for wall in self.v_walls:
            board_array[wall][3] = 1
        board_array[:, :, 4] = self.p1_walls
        board_array[:, :, 5] = self.p2_walls
        return board_array