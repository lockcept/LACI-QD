import argparse
import time
import tkinter as tk
from Board import Board
from Game import Game
from Players import Player, RandomPlayer, HumanPlayer, MCTSPlayer
from MCTS import MCTS
from Trainer import NNetWrapper
from utils import dotdict


class GUIQuoridor:
    def __init__(self, game):
        self.game = game
        self.board_size = game.n
        self.cell_size = 60
        self.wall_thickness = 10
        self.margin = 10
        self.root = tk.Tk()
        self.root.title("Quoridor Game")
        self.canvas = tk.Canvas(
            self.root,
            width=self.margin * 2 + self.cell_size * self.board_size,
            height=self.margin * 2 + self.cell_size * self.board_size,
            bg="white",
        )
        self.canvas.pack()

    def draw_board(self, board: Board):
        self.canvas.delete("all")

        for i in range(self.board_size + 1):
            x0 = self.margin + i * self.cell_size
            y0 = self.margin
            x1 = self.margin + i * self.cell_size
            y1 = self.margin + self.board_size * self.cell_size
            self.canvas.create_line(x0, y0, x1, y1, fill="black", width=1)

            x0 = self.margin
            y0 = self.margin + i * self.cell_size
            x1 = self.margin + self.board_size * self.cell_size
            y1 = self.margin + i * self.cell_size
            self.canvas.create_line(x0, y0, x1, y1, fill="black", width=1)

        # 벽 그리기
        for wall in board.h_walls:
            x, y = wall
            x_start = self.margin + y * self.cell_size
            y_start = self.margin + (x + 1) * self.cell_size - self.wall_thickness // 2
            x_end = x_start + self.cell_size * 2
            y_end = y_start + self.wall_thickness
            self.canvas.create_rectangle(
                x_start, y_start, x_end, y_end, fill="brown", outline="brown"
            )

        for wall in board.v_walls:  # 수직 벽
            x, y = wall
            x_start = self.margin + (y + 1) * self.cell_size - self.wall_thickness // 2
            y_start = self.margin + x * self.cell_size
            x_end = x_start + self.wall_thickness
            y_end = y_start + self.cell_size * 2  # 벽 길이 2
            self.canvas.create_rectangle(
                x_start, y_start, x_end, y_end, fill="brown", outline="brown"
            )

        # 플레이어 말 그리기
        player1_pos, player2_pos = board.p1_pos, board.p2_pos
        self.draw_piece(player1_pos, "red")
        self.draw_piece(player2_pos, "blue")

        self.root.update()

    def draw_piece(self, position, color):
        # x=0 위쪽, x=n 아래쪽으로 보정
        x, y = position
        x_center = self.margin + y * self.cell_size + self.cell_size // 2
        y_center = self.margin + x * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3
        self.canvas.create_oval(
            x_center - radius,
            y_center - radius,
            x_center + radius,
            y_center + radius,
            fill=color,
            outline=color,
        )


def parse_player(player_arg, game, mcts=None) -> Player:
    if player_arg == "human":
        return HumanPlayer(game)
    elif player_arg == "random":
        return RandomPlayer(game)
    elif player_arg == "mcts":
        nnet = NNetWrapper(game)
        nnet.load_checkpoint("./temp", "best.pth.tar")
        mcts = MCTS(
            game=game, nnet=nnet, args=dotdict({"numMCTSSims": 25, "cpuct": 1.0})
        )
        return MCTSPlayer(game, mcts)
    else:
        raise ValueError(f"Invalid player type: {player_arg}")


def play_game(
    player1: Player, player2: Player, game: Game, gui: GUIQuoridor, delay=0.2
):
    board = game.getInitBoard()
    cur_player = 1

    while True:
        gui.draw_board(board)

        if game.getGameEnded(board, cur_player) != None:
            result = game.getGameEnded(board, cur_player)
            if result == 1:
                print("Player 1 wins!")
            elif result == -1:
                print("Player 2 wins!")
            else:
                print("It's a draw!")
            break

        canonical_board = board.get_canonical_form(cur_player)

        if cur_player == 1:
            action = player1.play(canonical_board)
        else:
            action = player2.play(canonical_board)

        board, cur_player = game.getNextState(board, cur_player, action)

        if (
            delay
            and not isinstance(player1, HumanPlayer)
            and not isinstance(player2, HumanPlayer)
        ):
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Play a Quoridor game.")
    parser.add_argument(
        "--p1",
        type=str,
        required=True,
        choices=["human", "random", "mcts"],
        help="Type of player 1 (human, random, mcts)",
    )
    parser.add_argument(
        "--p2",
        type=str,
        required=True,
        choices=["human", "random", "mcts"],
        help="Type of player 2 (human, random, mcts)",
    )
    args = parser.parse_args()

    game = Game(n=9)

    gui = GUIQuoridor(game)

    player1 = parse_player(args.p1, game)
    player2 = parse_player(args.p2, game)

    play_game(player1, player2, game, gui, delay=0.2)


if __name__ == "__main__":
    main()
