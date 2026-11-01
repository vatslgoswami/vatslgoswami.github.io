---
title: Chess Self-Play with Reinforcement Learning
year: 2026
kind: Project
summary: An AlphaZero-style chess project combining exact endgame analysis, neural networks, search, and self-play reinforcement learning.
tags: [RL, Self-Play, Chess]
links:
  - { label: GitHub, url: "https://github.com/vatslgoswami/alphachess-RL" }
featured: true
detail: true
---

How much stronger does a chess agent become when it learns from games against itself? With Samyak and Agriya, I built a small-scale AlphaZero-style system to investigate how search and self-play reinforcement learning affect playing strength. The project connects exact solutions for small endgames with neural approximation and search in the much larger space of full chess.

## From exact solutions to learned play

Chess is a two-player, zero-sum game with far too many positions to solve exhaustively at the scale of this project. Small endgames offer a tractable starting point: with only a few pieces on the board, we can enumerate legal positions and work backward from terminal outcomes to determine optimal play. Those exact values provide a reference for studying the errors made by an approximate value function.

I worked on king-and-queen versus king (KQK), king-and-rook versus king (KRK), and king-and-pawn versus king (KPK). The endgame work included symmetry-reduced position indexing, retrograde analysis, value iteration, and comparisons with real-time dynamic programming. The resulting win/draw/loss classifications agreed with Syzygy tablebases on every legal position in the three solved endgames.

![Diagram showing endgame positions being solved and used to evaluate learned values and moves.](/images/projects/chess-endgames.svg)

The KRK experiments also compared learned value functions with the exact solution. This made it possible to examine not just prediction error, but whether a model's estimates supported good moves against a strong defender. These results concern the restricted endgame setting, rather than full-game Elo.

## Learning from games and search

The full-game system uses a policy–value network: the policy estimates promising moves, while the value estimates the outcome from a position. A supervised stage uses positions from Lichess games to provide an initial policy before reinforcement learning.

Monte Carlo tree search (MCTS) uses the network to guide exploration of candidate moves. Search produces a stronger training target than a single network prediction by combining the policy's preferences with lookahead. In self-play, the agent uses search to play both sides and records positions, search-derived move distributions, and game outcomes. The trainer samples those records to update the network, which in turn guides later search.

![Diagram showing supervised initialization followed by a loop of policy-value network, search, self-play, and reinforcement learning.](/images/projects/chess-self-play.svg)

## Working within a compute budget

We restricted self-play to positions from the French Defense to concentrate training on related middlegames. We also used endgame tablebases for positions with five or fewer pieces: self-play games reaching that boundary could be adjudicated using an exact result, and the playing system could use tablebase moves there. These choices made the task more manageable while narrowing the scope of the learned policy's evaluation; performance in the French Defense does not establish equal strength across every opening.

## Evaluation

The evaluation framework compares checkpoints under consistent search budgets and opening conditions, including colour-swapped games and matches against fixed-strength Stockfish opponents. It measures the effect of search separately from changes across self-play training. Endgame experiments use the solved positions as an independent reference for value estimates and move quality.

The central question is the change in playing strength from the supervised starting point to the self-play-trained agent. The public repository documents the project and its endgame results; further project work is maintained separately by the team.
