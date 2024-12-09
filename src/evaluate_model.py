"""
evaluate_model.py
"""

import csv
from pickle import Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from game import Game
from nnet_wrapper import NNetWrapper
from play_game import play_game
from players import Player, NNetPlayer, MctsPlayer, GreedyPlayer


def get_cross_entropy_from_greedy(model_file, train_examples):
    """
    Get the cross entropy and mean squared error from the greedy function.
    """
    game = Game(9)
    nnet_wrapper = NNetWrapper(game)
    nnet_wrapper.load_checkpoint("models", model_file)

    cross_entropy = 0
    score_mse = 0

    example_count = 10000
    print(len(train_examples))

    for i in tqdm(range(example_count), desc="Processing Examples"):
        board, pi, score = train_examples[i]
        pred_pi, pred_v = nnet_wrapper.predict(board)
        cross_entropy += -np.sum(pi * np.log(pred_pi + 1e-9))
        score_mse += (score - pred_v[0]) ** 2

    return cross_entropy / example_count, score_mse / example_count


def simulate_games(game: Game, player1: Player, player2: Player, game_num=5):
    """
    Play a game with the greedy function.
    """

    progress_bar = tqdm(range(game_num), desc="Playing Game with Greedy")

    player1_win_count = 0
    player2_win_count = 0
    draw_count = 0

    progress_bar = tqdm(range(game_num), desc="0 vs 0")

    for i in progress_bar:
        result = 0
        if i % 2 == 0:
            result = play_game(player1, player2, game, gui=None, delay=0)
        else:
            result = -play_game(player2, player1, game, gui=None, delay=0)

        if result == 1:
            player1_win_count += 1
        elif result == -1:
            player2_win_count += 1
        else:
            draw_count += 1

        progress_bar.set_description(
            f"{player1_win_count} vs {player2_win_count} | Draws: {draw_count}"
        )

    return player1_win_count / (player1_win_count + player2_win_count)


def main():
    csv_file_path = "logs/evaluation.csv"
    examples_path = "models/prepared.examples"

    with open(examples_path, "rb") as file:
        train_examples_history = Unpickler(file).load()

    train_examples = []
    for e in train_examples_history:
        train_examples.extend(e)
    shuffle(train_examples)

    game = Game(9)

    for i in range(0, 30):
        model_name = f"checkpoint_{i}.pth.tar"
        print(f"evaluating {model_name}")

        ce, mse = get_cross_entropy_from_greedy(model_name, train_examples)
        print(ce, mse)

        nnet_player = NNetPlayer(game, model_name)
        mcts_player = MctsPlayer(game, model_name)
        greedy_player = GreedyPlayer(game)

        nnet_win_rate_vs_greedy = simulate_games(game, nnet_player, greedy_player)
        mcts_win_rate_vs_greedy = simulate_games(game, mcts_player, greedy_player)
        mcts_win_rate_vs_nnet = simulate_games(game, mcts_player, nnet_player)

        with open(csv_file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(
                    [
                        "model_name",
                        "cross_entropy",
                        "score_MSE",
                        "nnet_win_rate_vs_greedy",
                        "mcts_win_rate_vs_greedy",
                        "mcts_win_rate_vs_nnet",
                    ]
                )
            writer.writerow(
                [
                    model_name,
                    ce,
                    mse,
                    nnet_win_rate_vs_greedy,
                    mcts_win_rate_vs_greedy,
                    mcts_win_rate_vs_nnet,
                ]
            )


if __name__ == "__main__":
    main()
