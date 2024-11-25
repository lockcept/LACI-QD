import json
import logging

from tqdm import tqdm

from game import Game

log = logging.getLogger(__name__)


class Arena:
    def __init__(self, player1, player2, game: Game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self):
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.get_init_board()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == None:
            it += 1
            action = players[curPlayer + 1](board.get_canonical_form(curPlayer))
            valids = self.game.getValidMoves(board.get_canonical_form(curPlayer))
            if valids[action] == 0:
                log.error(f"Action {action} is not valid!")
                log.debug(f"valids = {valids}")
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num):
        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        game_histories = []
        winningCriteria = self.game.winning_criteria
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame()
            game_histories.append(gameResult)
            if gameResult > winningCriteria:
                oneWon += 1
            elif gameResult < -winningCriteria:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame()
            game_histories.append(-gameResult)
            if gameResult < -winningCriteria:
                oneWon += 1
            elif gameResult > winningCriteria:
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
