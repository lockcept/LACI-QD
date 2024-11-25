"""
Arena class to run games between two players.
"""

import csv
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from game import Game

log = logging.getLogger(__name__)


class Arena:
    """
    Class to run games between two players.
    """

    def __init__(self, player1, player2, game: Game):
        self.player1 = player1
        self.player2 = player2
        self.game = game

    def play_game(self, swapped=False):
        """
        Plays a single game. If `swapped` is True, player1 and player2 are swapped.
        """
        players = (
            [self.player2, None, self.player1]
            if not swapped
            else [self.player1, None, self.player2]
        )
        cur_player = 1
        board = self.game.get_init_board()
        it = 0
        while self.game.get_win_status(board, cur_player) is None:
            it += 1
            action = players[cur_player + 1](board.get_canonical_form(cur_player))
            valids = self.game.get_valid_actions(board.get_canonical_form(cur_player))
            if valids[action] == 0:
                log.error(f"Action {action} is not valid!")
                log.debug(f"valids = {valids}")
                assert valids[action] > 0
            board, cur_player = self.game.get_next_state(board, cur_player, action)

        result = cur_player * self.game.get_win_status(board, cur_player)
        return -result if swapped else result

    def _play_single_game(self, args):
        """
        Helper function for multiprocessing.
        `args` contains whether to swap players and the game index.
        """
        swapped, _ = args
        return self.play_game(swapped=swapped)

    def play_games(self, num, num_iter):
        """
        Plays `num` games in parallel using multiprocessing.
        """
        num = int(num / 2)
        game_args = [(False, i) for i in range(num)] + [(True, i) for i in range(num)]

        # Use multiprocessing Pool to parallelize game playing
        with Pool(processes=cpu_count() // 2) as pool:
            results = list(
                tqdm(
                    pool.imap(self._play_single_game, game_args),
                    total=num * 2,
                    desc="Arena.playGames",
                )
            )

        # Process results
        player1_win = 0
        player2_win = 0
        draws = 0
        game_histories = []
        winning_criteria = self.game.winning_criteria

        for game_result in results:
            game_histories.append(game_result)
            if game_result > winning_criteria:
                player1_win += 1
            elif game_result < -winning_criteria:
                player2_win += 1
            else:
                draws += 1

        # Round float values to 3 decimal places
        rounded_game_histories = [round(float(x), 3) for x in game_histories]

        with open("logs/arena.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if f.tell() == 0:
                writer.writerow(["num_iter", "num_arena", "score"])

            # Write rows
            for index, score in enumerate(rounded_game_histories):
                writer.writerow([num_iter, index, score])

        return player1_win, player2_win, draws
