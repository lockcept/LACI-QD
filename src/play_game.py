"""
This module allows users to simulate a game of Quoridor between two players using various strategies.
"""

import argparse
import time
from gui import GUIQuoridor
from game import Game
from players import (
    GreedyMctsPlayer,
    GreedyPlayer,
    Player,
    RandomPlayer,
    HumanPlayer,
    MctsPlayer,
)
from mcts import MCTS
from nnet_wrapper import NNetWrapper
from utils import Docdict


def play_game(
    player1: Player, player2: Player, game: Game, gui: GUIQuoridor, delay=0.2
):
    """
    Simulates a game between two players using the provided game logic and GUI.

    Args:
        player1 (Player): The first player.
        player2 (Player): The second player.
        game (Game): The game logic handler.
        gui (GUIQuoridor): The graphical user interface for the game.
        delay (float, optional): The delay between moves in seconds. Defaults to 0.2.

    Returns:
        None
    """
    board = game.get_init_board()
    cur_player = 1
    gui.update_board(board)

    while True:
        if game.get_win_status(board, cur_player) is not None:
            result = game.get_win_status(board, cur_player)
            if result == 1:
                print("Player 1 wins!")
            elif result == -1:
                print("Player 2 wins!")
            else:
                print("It's a draw!")
            break

        canonical_board = board.get_canonical_form(cur_player)

        if cur_player == 1:
            action, probabilities = player1.play(canonical_board, reverse_x=False)
        else:
            action, probabilities = player2.play(canonical_board, reverse_x=True)
        gui.is_human_turn = False

        gui.update_board(board, action_probabilities=probabilities, player=cur_player)

        if probabilities is not None and delay:
            time.sleep(delay)
            gui.update_board(board)
            time.sleep(delay / 3)

        board, cur_player = game.get_next_state(board, cur_player, action)

        gui.update_board(board)

        if delay:
            time.sleep(delay)


def main():
    """
    Main function to parse command line arguments and start a game.
    """
    parser = argparse.ArgumentParser(description="Play a Quoridor game.")
    choices = ["human", "random", "mcts", "greedy", "greedymcts"]
    parser.add_argument(
        "--p1",
        type=str,
        required=True,
        choices=choices,
        help="Type of player 1",
    )
    parser.add_argument(
        "--p2",
        type=str,
        required=True,
        choices=choices,
        help="Type of player 2",
    )
    args = parser.parse_args()

    game = Game(n=9)

    gui = GUIQuoridor(game)

    def parse_player(player_arg, game) -> Player:
        if player_arg == "human":
            return HumanPlayer(game, gui=gui)
        elif player_arg == "random":
            return RandomPlayer(game)
        elif player_arg == "mcts":
            nnet_wrapper = NNetWrapper(game)
            nnet_wrapper.load_checkpoint("./models", "best.pth.tar")
            mcts = MCTS(
                game=game,
                pi_v_function=nnet_wrapper.get_pi_v,
                args=Docdict({"numMCTSSims": 25, "cpuct": 1.0}),
            )
            return MctsPlayer(game, mcts)
        elif player_arg == "greedy":
            return GreedyPlayer(game)
        elif player_arg == "greedymcts":
            return GreedyMctsPlayer(game)
        else:
            raise ValueError(f"Invalid player type: {player_arg}")

    player1 = parse_player(args.p1, game)
    player2 = parse_player(args.p2, game)

    play_game(player1, player2, game, gui, delay=0.1)


if __name__ == "__main__":
    main()
