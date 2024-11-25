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
        self.my_pos = (0, n // 2)
        self.enemy_pos = (n - 1, n // 2)
        self.my_walls = n * n // 8
        self.enemy_walls = n * n // 8
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
            board.my_pos = self.my_pos
            board.enemy_pos = self.enemy_pos
            board.my_walls = self.my_walls
            board.enemy_walls = self.enemy_walls
            board.h_walls = set([(x, y) for x, y in self.h_walls])
            board.v_walls = set([(x, y) for x, y in self.v_walls])
            board.turns = self.turns
            return board
        else:
            board = Board(self.n)
            board.my_pos = (self.n - 1 - self.enemy_pos[0], self.enemy_pos[1])
            board.enemy_pos = (self.n - 1 - self.my_pos[0], self.my_pos[1])
            board.my_walls = self.enemy_walls
            board.enemy_walls = self.my_walls
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
        board.my_pos = (self.my_pos[0], self.n - 1 - self.my_pos[1])
        board.enemy_pos = (self.enemy_pos[0], self.n - 1 - self.enemy_pos[1])
        board.my_walls = self.my_walls
        board.enemy_walls = self.enemy_walls
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
            f"({self.my_pos[0]},{self.my_pos[1]})"
            + ":"
            + f"({self.enemy_pos[0]},{self.enemy_pos[1]})"
            + ":"
            + format_sorted_tuple_set(self.h_walls)
            + ":"
            + format_sorted_tuple_set(self.v_walls)
            + ":"
            + str(self.my_walls)
            + ":"
            + str(self.enemy_walls)
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
        self.my_pos = move
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

        self.my_walls -= 1
        self.turns += 1

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
        pos = self.my_pos
        opponent_pos = self.enemy_pos

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

        if (
            self.get_distance_to_goal(1, temp_h_walls, temp_v_walls) == -1
            or self.get_distance_to_goal(-1, temp_h_walls, temp_v_walls) == -1
        ):
            return False

        return True

    def get_distance_to_goal(self, player, h_walls, v_walls):
        """
        Determines the minimum distance for a player to reach their goal row on the board.

        Args:
            start (tuple): The starting position (x, y) of the player.
            player (int): The player number (1 or -1). Player 1 aims for the last row, player -1 aims for the first row.
            h_walls (set): A set of tuples representing the positions of horizontal walls.
            v_walls (set): A set of tuples representing the positions of vertical walls.

        Returns:
            int: The minimum distance to the goal row if reachable, otherwise -1.
        """
        goal_row = self.n - 1 if player == 1 else 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        start = self.my_pos if player == 1 else self.enemy_pos
        queue = deque([(start, 0)])  # (current_position, distance)
        visited = set()
        visited.add(start)

        while queue:
            (x, y), distance = queue.popleft()

            # Check if the current position is on the goal row
            if x == goal_row:
                return distance

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.n and 0 <= ny < self.n and (nx, ny) not in visited:
                    # Check wall conditions
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
                        queue.append(((nx, ny), distance + 1))
                        visited.add((nx, ny))

        # If the goal row is not reachable, return -1
        return -1

    def get_legal_walls(self):
        """
        Returns a list of legal walls for player 1
        """
        legal_walls = []
        if self.my_walls <= 0:
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
        board_array[self.my_pos][0] = 1
        board_array[self.enemy_pos][1] = 1
        for wall in self.h_walls:
            board_array[wall][2] = 1
        for wall in self.v_walls:
            board_array[wall][3] = 1
        board_array[:, :, 4] = self.my_walls
        board_array[:, :, 5] = self.enemy_walls
        return board_array

    def display(self):
        """
        Displays the current state of the board.

        Returns:
            None
        """
        n = self.n
        board_size_with_wall = 2 * n - 1
        display_board = np.full(
            (board_size_with_wall, board_size_with_wall), " ", dtype=str
        )

        for x in range(n):
            for y in range(n):
                display_board[x * 2, y * 2] = "□"

        display_board[self.my_pos[0] * 2, self.my_pos[1] * 2] = "●"
        display_board[self.enemy_pos[0] * 2, self.enemy_pos[1] * 2] = "■"

        for x, y in self.h_walls:
            display_board[x * 2 + 1, y * 2] = "━"
            display_board[x * 2 + 1, y * 2 + 1] = "━"
            display_board[x * 2 + 1, y * 2 + 2] = "━"
        for x, y in self.v_walls:
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
            self.my_walls,
            ", ",
            "wall 2: ",
            self.enemy_walls,
        )
