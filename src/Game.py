import numpy as np
from Board import Board


class Game:
    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        b = Board(self.n)
        return b.to_array()

    def getBoardSize(self):
        return (
            self.n,
            self.n,
            7,
        )  # p1_pos, p2_pos, h_walls, v_walls, p1_walls, p2_walls, turns

    def getActionSize(self):
        return self.n * self.n + (self.n - 1) * (self.n - 1) * 2

    def getNextState(self, board, player: int, action: int):
        # valid action만 들어온다고 가정
        board = self.getCanonicalForm(board, player)

        b = Board(self.n)
        b.p1_pos = tuple(np.argwhere(board[:, :, 0] == 1)[0])
        b.p2_pos = tuple(np.argwhere(board[:, :, 1] == 1)[0])
        b.p1_walls = int(board[0, 0, 4])
        b.p2_walls = int(board[0, 0, 5])
        b.turn_count = int(board[0, 0, 6])

        for i in range(self.n):
            for j in range(self.n):
                if board[i, j, 2] == 1:
                    b.h_walls.add((i, j))
                if board[i, j, 3] == 1:
                    b.v_walls.add((i, j))

        if action < self.n * self.n:
            move = (action // self.n, action % self.n)
            b.execute_move(move, 1)
        else:
            wall_type = (action - self.n * self.n) // ((self.n - 1) * (self.n - 1))
            wall_pos = (action - self.n * self.n) % ((self.n - 1) * (self.n - 1))
            wall_pos = (wall_pos // (self.n - 1), wall_pos % (self.n - 1))
            b.place_wall(wall_pos, wall_type, 1)

        canonical_b = self.getCanonicalForm(b.to_array(), player)
        return (canonical_b, -player)

    def getValidMoves(self, board, player):
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.p1_pos = tuple(np.argwhere(board[:, :, 0] == 1)[0])
        b.p2_pos = tuple(np.argwhere(board[:, :, 1] == 1)[0])
        b.p1_walls = int(board[0, 0, 4])
        b.p2_walls = int(board[0, 0, 5])

        for i in range(self.n):
            for j in range(self.n):
                if board[i, j, 2] == 1:
                    b.h_walls.add((i, j))
                if board[i, j, 3] == 1:
                    b.v_walls.add((i, j))

        # 이동 가능 체크
        moves = b.get_legal_moves(player)
        for move in moves:
            valids[move[0] * self.n + move[1]] = 1

        # 벽 설치 가능 체크
        walls = b.get_legal_walls(player)
        for wall in walls:
            wall_idx = self.n * self.n + wall[0] * (self.n - 1) + wall[1]
            if wall[2] == "h":
                wall_idx += 0
            else:
                wall_idx += (self.n - 1) * (self.n - 1)
            valids[wall_idx] = 1

        return np.array(valids)

    def getGameEnded(self, board, player):
        b = Board(self.n)
        b.p1_pos = tuple(np.argwhere(board[:, :, 0] == 1)[0])
        b.p2_pos = tuple(np.argwhere(board[:, :, 1] == 1)[0])
        b.p1_walls = int(board[0, 0, 4])
        b.p2_walls = int(board[0, 0, 5])
        b.turn_count = int(board[0, 0, 6])

        for i in range(self.n):
            for j in range(self.n):
                if board[i, j, 2] == 1:
                    b.h_walls.add((i, j))
                if board[i, j, 3] == 1:
                    b.v_walls.add((i, j))

        if b.turn_count == 0:
            return -0.1

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        if player == 1:
            return board
        else:
            new_board = np.zeros_like(board)
            new_board[:, :, 0:2] = np.flip(board[:, :, 0:2], axis=[0, 2])
            new_board[:-1, :-1, 2:4] = np.flip(board[:-1, :-1, 2:4], axis=0)
            new_board[:, :, 4:6] = np.flip(board[:, :, 4:6], axis=2)
            new_board[:, :, 6:7] = np.copy(board[:, :, 6:7])
            return new_board

    def getFlipedForm(self, board):
        new_board = np.zeros_like(board)
        new_board[:, :, 0:2] = np.flip(board[:, :, 0:2], axis=1)
        new_board[:-1, :-1, 2:4] = np.flip(board[:-1, :-1, 2:4], axis=1)
        new_board[:, :, 4:6] = np.copy(board[:, :, 4:6])
        new_board[:, :, 6:7] = np.copy(board[:, :, 6:7])
        return new_board

    def getSymmetries(self, board, pi):
        assert len(pi) == self.getActionSize()

        n = self.n
        pi_board = np.reshape(pi[: n * n], (n, n))
        pi_walls = np.reshape(pi[n * n :], (2, n - 1, n - 1))

        symmetries = []

        # 원래 모습
        symmetries.append((board, pi))

        # 좌우 대칭
        board_lr = self.getFlipedForm(board)
        pi_board_lr = np.fliplr(pi_board)
        pi_walls_lr = np.flip(pi_walls, axis=2)
        pi_lr = list(pi_board_lr.ravel()) + list(pi_walls_lr.ravel())
        symmetries.append((board_lr, pi_lr))

        return symmetries

    def stringRepresentation(self, board):
        # 보드 상태를 문자열로 반환
        return board.tostring()

    @staticmethod
    def display(board):
        n = board.shape[0]
        board_size = 2 * n - 1
        display_board = np.full((board_size, board_size), " ", dtype=str)

        for x in range(n):
            for y in range(n):
                piece = board[x, y, :]
                dx, dy = 2 * x, 2 * y

                if piece[0] == 1:
                    display_board[dx, dy] = "1"
                elif piece[1] == 1:
                    display_board[dx, dy] = "2"
                else:
                    display_board[dx, dy] = "."

                if piece[2] == 1:
                    display_board[dx + 1, dy] = "-"
                    display_board[dx + 1, dy + 1] = "-"
                    display_board[dx + 1, dy + 2] = "-"
                if piece[3] == 1:
                    display_board[dx, dy + 1] = "|"
                    display_board[dx + 1, dy + 1] = "|"
                    display_board[dx + 2, dy + 1] = "|"

        print("  ", end="")
        for y in range(board_size):
            print(y % 10, end=" ")
        print("")

        for x in range(board_size):
            print(x % 10, end=" ")
            for y in range(board_size):
                print(display_board[x, y], end=" ")
            print("")
        print(
            "wall 1: ",
            board[0, 0, 4],
            ", wall 2: ",
            board[0, 0, 5],
            ", turn count: ",
            board[0, 0, 6],
        )
