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
        self.q_sa = {}  # stores Q values for s,a (as defined in the paper)
        self.n_sa = {}  # stores #times edge s,a was visited
        self.n_s = {}  # stores #times board s was visited
        self.p_s = {}  # stores initial policy (returned by pi_v_function)

        self.e_s = {}  # stores game.get_win_status ended for board s
        self.v_s = {}  # stores game.get_valid_actions for board s

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
            self.n_sa[(s, a)] if (s, a) in self.n_sa else 0
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

        if s not in self.e_s:
            self.e_s[s] = self.game.get_win_status(board, 1)
        if self.e_s[s] is not None:
            # terminal node
            return -self.e_s[s]

        if s not in self.p_s:
            # leaf node
            self.p_s[s], v = self.pi_v_function(board)  # Use the passed function
            valids = self.game.get_valid_actions(board)
            self.p_s[s] = self.p_s[s] * valids  # masking invalid moves
            sum_p_s = np.sum(self.p_s[s])
            if sum_p_s > 0:
                self.p_s[s] /= sum_p_s  # renormalize
            else:
                # If all valid moves were masked, make all valid moves equally probable.
                log.error("All valid moves were masked, doing a workaround.")
                board.display()
                self.p_s[s] = self.p_s[s] + valids
                self.p_s[s] /= np.sum(self.p_s[s])

            self.v_s[s] = valids
            self.n_s[s] = 0
            return -v

        valids = self.v_s[s]
        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.q_sa:
                    u = self.q_sa[(s, a)] + self.args.cpuct * self.p_s[s][
                        a
                    ] * math.sqrt(self.n_s[s]) / (1 + self.n_sa[(s, a)])
                else:
                    u = (
                        self.args.cpuct * self.p_s[s][a] * math.sqrt(self.n_s[s] + EPS)
                    )  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.get_next_state(board, 1, a)
        next_s = next_s.get_canonical_form(next_player)

        v = self.search(next_s)

        if (s, a) in self.q_sa:
            self.q_sa[(s, a)] = (self.n_sa[(s, a)] * self.q_sa[(s, a)] + v) / (
                self.n_sa[(s, a)] + 1
            )
            self.n_sa[(s, a)] += 1

        else:
            self.q_sa[(s, a)] = v
            self.n_sa[(s, a)] = 1

        self.n_s[s] += 1
        return -v
