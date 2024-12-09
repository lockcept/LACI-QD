"""
This module allows users to simulate a game of Quoridor between two players using various strategies.
"""

import argparse
import time
from typing import Optional

from tqdm import tqdm
from gui import GUIQuoridor
from game import Game
from players import (
    GreedyMctsPlayer,
    GreedyPlayer,
    NNetPlayer,
    Player,
    RandomPlayer,
    HumanPlayer,
    MctsPlayer,
)


def play_game(
    player1: Player,
    player2: Player,
    game: Game,
    gui: Optional[GUIQuoridor] = None,
    delay=0.2,
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

    if gui is not None:
        gui.update_board(board)

    while True:
        if game.get_win_status(board, 1) is not None:
            result = game.get_win_status(board, 1)
            if result == 1:
                return 1
            elif result == -1:
                return -1
            else:
                return 0
            break

        canonical_board = board.get_canonical_form(cur_player)

        if cur_player == 1:
            action, probabilities = player1.play(canonical_board, reverse_x=False)
        else:
            action, probabilities = player2.play(canonical_board, reverse_x=True)

        if gui is not None:
            gui.is_human_turn = False
            gui.update_board(
                board, action_probabilities=probabilities, player=cur_player
            )

        if probabilities is not None and delay:
            time.sleep(delay)
            gui.update_board(board)
            time.sleep(delay / 3)

        board, cur_player = game.get_next_state(board, cur_player, action)

        if gui is not None:
            gui.update_board(board)

        if delay:
            time.sleep(delay)


def main():
    """
    Main function to parse command line arguments and start a game.
    """
    parser = argparse.ArgumentParser(description="Play a Quoridor game.")
    choices = ["human", "random", "mcts", "nnet", "greedy", "greedymcts"]
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
    parser.add_argument(
        "--gui",
        action="store_true",
        required=False,
        default=False,
    )
    args = parser.parse_args()

    game = Game(n=9)

    gui = GUIQuoridor(game) if args.gui else None

    def parse_player(player_arg, game) -> Player:
        if player_arg == "human":
            return HumanPlayer(game, gui=gui)
        elif player_arg == "random":
            return RandomPlayer(game)
        elif player_arg == "mcts":
            return MctsPlayer(game)
        elif player_arg == "nnet":
            return NNetPlayer(game)
        elif player_arg == "greedy":
            return GreedyPlayer(game)
        elif player_arg == "greedymcts":
            return GreedyMctsPlayer(game)
        else:
            raise ValueError(f"Invalid player type: {player_arg}")

    player1 = parse_player(args.p1, game)
    player2 = parse_player(args.p2, game)

    battle_counts = 100
    player1_win_count = 0
    player2_win_count = 0
    first_player_win_count = 0
    draw_count = 0

    progress_bar = tqdm(range(battle_counts), desc=f"{args.p1}_0 vs {args.p2}_0")

    for i in progress_bar:
        result = 0
        if i % 2 == 0:
            result = play_game(player1, player2, game, gui, delay=0)
        else:
            result = -play_game(player2, player1, game, gui, delay=0)

        if result == 1:
            player1_win_count += 1
        elif result == -1:
            player2_win_count += 1
        else:
            draw_count += 1

        if i % 2 == 0:
            if result == 1:
                first_player_win_count += 1
        else:
            if result == -1:
                first_player_win_count += 1

        progress_bar.set_description(
            f"{args.p1}_{player1_win_count} vs {args.p2}_{player2_win_count} | Draws: {draw_count} | First wins: {first_player_win_count}"
        )


if __name__ == "__main__":
    main()
