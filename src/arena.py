import json
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from game import Game

log = logging.getLogger(__name__)


class Arena:
    def __init__(self, player1, player2, game: Game):
        self.player1 = player1
        self.player2 = player2
        self.game = game

    def playGame(self, swapped=False):
        """
        Plays a single game. If `swapped` is True, player1 and player2 are swapped.
        """
        players = (
            [self.player2, None, self.player1]
            if not swapped
            else [self.player1, None, self.player2]
        )
        curPlayer = 1
        board = self.game.get_init_board()
        it = 0
        while self.game.get_win_status(board, curPlayer) == None:
            it += 1
            action = players[curPlayer + 1](board.get_canonical_form(curPlayer))
            valids = self.game.get_valid_actions(board.get_canonical_form(curPlayer))
            if valids[action] == 0:
                log.error(f"Action {action} is not valid!")
                log.debug(f"valids = {valids}")
                assert valids[action] > 0
            board, curPlayer = self.game.get_next_state(board, curPlayer, action)

        result = curPlayer * self.game.get_win_status(board, curPlayer)
        return -result if swapped else result

    def _play_single_game(self, args):
        """
        Helper function for multiprocessing.
        `args` contains whether to swap players and the game index.
        """
        swapped, _ = args
        return self.playGame(swapped=swapped)

    def playGames(self, num):
        """
        Plays `num` games in parallel using multiprocessing.
        """
        num = int(num / 2)
        game_args = [(False, i) for i in range(num)] + [(True, i) for i in range(num)]

        # Use multiprocessing Pool to parallelize game playing
        with Pool(processes=cpu_count() - 3) as pool:
            results = list(
                tqdm(
                    pool.imap(self._play_single_game, game_args),
                    total=num * 2,
                    desc="Arena.playGames",
                )
            )

        # Process results
        oneWon = 0
        twoWon = 0
        draws = 0
        game_histories = []
        winningCriteria = self.game.winning_criteria

        for gameResult in results:
            game_histories.append(gameResult)
            if gameResult > winningCriteria:
                oneWon += 1
            elif gameResult < -winningCriteria:
                twoWon += 1
            else:
                draws += 1

        # Round float values to 3 decimal places
        rounded_game_histories = [round(float(x), 3) for x in game_histories]

        # Save to file as pretty JSON
        with open("logs/arena.txt", "a") as f:
            json.dump(rounded_game_histories, f, indent=4)
            f.write("\n")

        return oneWon, twoWon, draws
