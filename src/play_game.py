import argparse
import time
from GUI import GUIQuoridor
from Game import Game
from Players import Player, RandomPlayer, HumanPlayer, MCTSPlayer
from MCTS import MCTS
from Trainer import NNetWrapper
from utils import dotdict


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

    def parse_player(player_arg, game) -> Player:
        if player_arg == "human":
            return HumanPlayer(game, gui=gui)
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

    player1 = parse_player(args.p1, game)
    player2 = parse_player(args.p2, game)

    play_game(player1, player2, game, gui, delay=0.2)


if __name__ == "__main__":
    main()
