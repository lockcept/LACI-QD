"""
This module implements the Monte Carlo Tree Search (MCTS) algorithm for game playing.
"""

import logging
import math
import numpy as np

from board import Board
from game import Game

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    """
    Monte Carlo Tree Search.
    """

    def __init__(self, game: Game, pi_v_function, args):
        """
        Initializes the MCTS with the given game and a pi_v_function.

        Args:
            game (Game): The game for which MCTS is applied.
            pi_v_function (function): A function that takes a board state as input
                                      and returns (pi, v), where:
                                      - pi: A policy vector representing action probabilities.
                                      - v: A value representing the board state's evaluation.
            args: Additional arguments for MCTS settings.
        """
        self.game = game
        self.pi_v_function = pi_v_function
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by pi_v_function)

        self.Es = {}  # stores game.get_win_status ended for board s
        self.Vs = {}  # stores game.get_valid_actions for board s

    def get_action_prob(self, board: Board, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for _ in range(self.args.numMCTSSims):
            self.search(board)

        s = board.string_representation()
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.get_action_size())
        ]

        if temp == 0:
            best_actions = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs = [0] * len(counts)
            probs[best_action] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, board: Board):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the pi_v_function is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = board.string_representation()

        if s not in self.Es:
            self.Es[s] = self.game.get_win_status(board, 1)
        if self.Es[s] is not None:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.pi_v_function(board)  # Use the passed function
            valids = self.game.get_valid_actions(board)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # If all valid moves were masked, make all valid moves equally probable.
                log.error("All valid moves were masked, doing a workaround.")
                board.display()
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(
                        self.Ns[s]
                    ) / (1 + self.Nsa[(s, a)])
                else:
                    u = (
                        self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                    )  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.get_next_state(board, 1, a)
        next_s = next_s.get_canonical_form(next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
