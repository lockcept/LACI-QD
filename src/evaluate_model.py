"""
evaluate_model.py
"""

import csv
from pickle import Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp

from game import Game
from nnet_wrapper import NNetWrapper
from play_game import play_game
from players import NNetPlayer, MctsPlayer, GreedyPlayer


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


def parallel_play_game(args):
    """
    play game function for multiprocessing
    """
    i, player1, player2, game = args
    if i % 2 == 0:
        return play_game(player1, player2, game, gui=None, delay=0)
    else:
        return -play_game(player2, player1, game, gui=None, delay=0)


def run_games_with_multiprocessing(
    player1_type, player2_type, model_name, game: Game, game_num=100, num_processes=8
):
    """
    Run games with multiprocessing
    if num_processes is 1, it will run in single process
    """
    player1_win_count = 0
    player2_win_count = 0
    draw_count = 0

    args = []
    for i in range(game_num):
        if player1_type == "mcts":
            player1 = MctsPlayer(game, model_name)
        elif player1_type == "nnet":
            player1 = NNetPlayer(game, model_name)
        else:
            player1 = GreedyPlayer(game)

        if player2_type == "mcts":
            player2 = MctsPlayer(game, model_name)
        elif player2_type == "nnet":
            player2 = NNetPlayer(game, model_name)
        else:
            player2 = GreedyPlayer(game)

        args.append((i, player1, player2, game))

    if num_processes == 1:
        results = []
        for arg in tqdm(args, total=len(args), desc="Self Play"):
            result = parallel_play_game(arg)
            results.append(result)
    else:
        with mp.Pool(processes=num_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(parallel_play_game, args),
                    total=len(args),
                    desc="Self Play",
                )
            )

    for game_result in results:
        if game_result == 1:
            player1_win_count += 1
        elif game_result == -1:
            player2_win_count += 1
        else:
            draw_count += 1

    print(player1_win_count, player2_win_count, draw_count)

    return player1_win_count / (player1_win_count + player2_win_count)


def main():
    """
    Main function to evaluate the model.
    """

    csv_file_path = "logs/evaluation.csv"
    examples_path = "examples/prepared.examples"

    with open(examples_path, "rb") as file:
        train_examples_history = Unpickler(file).load()

    train_examples = []
    for e in train_examples_history:
        train_examples.extend(e)
    shuffle(train_examples)

    game = Game(9)

    for i in range(1, 100, 10):
        model_name = f"checkpoint_{i}.pth.tar"
        print(f"evaluating {model_name}")

        ce, mse = get_cross_entropy_from_greedy(model_name, train_examples)
        print(ce, mse)

        nnet_win_rate_vs_greedy = run_games_with_multiprocessing(
            "nnet", "greedy", model_name, game
        )
        mcts_win_rate_vs_greedy = run_games_with_multiprocessing(
            "mcts", "greedy", model_name, game
        )
        mcts_win_rate_vs_nnet = run_games_with_multiprocessing(
            "mcts", "nnet", model_name, game
        )

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
    mp.set_start_method("spawn", force=True)
    main()
