# LACI-QD

Lockcept AI of Quoridor

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)

## Overview

LACI-QD is an artificial intelligence project for the game Quoridor, inspired by the AlphaGo Zero approach. It is designed to learn and execute high-level strategies in the game of Quoridor by combining reinforcement learning and Monte Carlo Tree Search (MCTS) techniques.

## Installation

To set up the environment, please follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/LACI-QD.git
cd LACI-QD

# Create and activate the conda environment
conda create --name LACI python=3.10
conda activate LACI

# Install dependencies
pip install -r requirements.txt
```

To deactivate the Conda environment after usage:

```bash
conda deactivate
```

## Usage

### Train the MCTS Model

To train the MCTS model, run the following command:

```bash
python src/main.py
```

### Play the Quoridor Game

To play the Quoridor game, use the following command:

```bash
python src/play_game.py --p1 [player_type] --p2 [player_type]
```

### [player_type] Options

- human: Play as a human player.
- random: A player making random moves.
- mcts: A player powered by the MCTS model.

Example:

```bash
python src/play_game.py --p1 human --p2 mcts
```
