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


def get_cross_entropy_from_greedy(model_file, examples_path):
    """
    Get the cross entropy and mean squared error from the greedy function.
    """
    game = Game(9)
    nnet_wrapper = NNetWrapper(game)
    nnet_wrapper.load_checkpoint("models", model_file)

    with open(examples_path, "rb") as file:
        train_examples_history = Unpickler(file).load()

    train_examples_count = 0
    for e in train_examples_history:
        train_examples_count += len(e)

    print("total examples: ", train_examples_count)

    train_examples = []
    for e in train_examples_history:
        train_examples.extend(e)
    shuffle(train_examples)

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


if __name__ == "__main__":
    MODEL_NAME = "temp.pth.tar"
    EXAMPLES_NAME = "prepared.examples"
    CSV_FILE_PATH = "logs/evaluation.csv"

    EXAMPLES_PATH = f"models/{EXAMPLES_NAME}"

    CE, MSE = get_cross_entropy_from_greedy(MODEL_NAME, EXAMPLES_PATH)
    print(CE, MSE)

    with open(CSV_FILE_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(
                ["model_name", "examples_name", "cross_entropy", "score_MSE"]
            )
        writer.writerow([MODEL_NAME, EXAMPLES_NAME, CE, MSE])
