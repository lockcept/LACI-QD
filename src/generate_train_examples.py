"""
generate train examples by simulating the game with GreedyPlayer
"""

import os
from pickle import Pickler

import numpy as np
from tqdm import tqdm
from game import Game
from players import greedy_function


def generate_train_examples(game: Game, num_games: int):
    """
    Simulates a game between two players and return the examples.
    """
    board = game.get_init_board()
    cur_player = 1
    examples = []

    for _ in tqdm(range(num_games), desc="Generating examples"):
        board = game.get_init_board()
        cur_player = 1

        while True:
            canonical_board = board.get_canonical_form(cur_player)
            pi, _ = greedy_function(game, canonical_board)
            score = game.get_win_status(canonical_board, 1, force_finish=True)
            sym = game.get_symmetries(canonical_board, pi)

            for b, p in sym:
                examples.append([game.board_to_input(b), p, score])

            if game.get_win_status(board, 1) is not None:
                break

            action = np.random.choice(len(pi), p=pi)

            board, cur_player = game.get_next_state(board, cur_player, action)

    return examples


if __name__ == "__main__":
    g = Game(n=9)
    train_examples_history = []

    FOLDER = "models"

    for i in range(20):
        print(f"Generating train examples {i+1}")
        train_examples = generate_train_examples(g, 100)
        train_examples_history.append(train_examples)

    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
    filename = os.path.join(FOLDER, "prepared.examples")
    with open(filename, "wb+") as f:
        Pickler(f).dump(train_examples_history)
    f.close()
