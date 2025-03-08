# Tic Tac Toe RL

A Tic Tac Toe game with advanced Reinforcement Learning (RL) agents. The project features both traditional Q-learning and Deep Q-learning implementations, with a graphical user interface where you can watch agents train and play against them.

## Features

- Play Tic Tac Toe against trained RL agents (Q-learning or Deep Q-learning)
- Train agents in real-time with visual feedback
- Choose between playing as X or O
- Select different opponent types for training (Random, MiniMax, or Self-Play)
- Performance metrics and visualization of training progress
- Hyperparameter tuning capabilities
- Experience replay for more stable learning
- Self-play training mode for advanced strategies
- Save and load trained agent models
- Real-time visualization of training progress and win rates

## Requirements

- Python 3.6+
- NumPy
- Pygame
- TensorFlow
- Matplotlib

## Installation

1. Clone this repository
2. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

### GUI Mode

Run the main script to start the game with GUI:

```
python main.py
```

### Training Mode

Train an agent from the command line with various options:

```
python main.py --train --agent deep --episodes 50000 --opponent self --visualize
```

Options:

- `--train`: Enable training mode
- `--agent [q|deep]`: Select agent type (Q-learning or Deep Q-learning)
- `--episodes N`: Number of training episodes
- `--opponent [random|minimax|self]`: Type of opponent to train against
- `--visualize`: Show training metrics after completion
- `--save-path PATH`: Custom path to save the trained agent
- `--load-path PATH`: Path to load an existing agent

### Self-Play Training

For more advanced training using self-play:

```
python self_play.py --agent deep --episodes 100000
```

### Hyperparameter Tuning

Find the optimal hyperparameters for your agent:

```
python hyperparameter_tuning.py
```

## Controls (GUI mode)

- Click on the board to place your symbol when it's your turn
- Use the "Train Agent" button to start training the RL agent
- Press ESC to stop training
- Use "Reset Game" to start a new game
- "Save Agent" and "Load Agent" buttons allow you to save and load trained models
- Use the radio buttons to select your preferred opponent type and whether you want to play as X or O

## Agent Types

### Traditional Q-learning Agent

- Table-based approach storing Q-values for each state-action pair
- Works well for simple games like Tic Tac Toe
- Fast training but limited generalization

### Deep Q-learning Agent

- Neural network approximates the Q-function
- Better generalization to similar states
- Experience replay for more stable learning
- Target network for more stable Q-value estimates
- Double DQN implementation to reduce overestimation

## Training Methods

### Against Random Opponent

- Good for learning basic strategies
- Fast training but limited skill ceiling

### Against MiniMax Opponent

- Learns more defensive and optimal play
- Challenging opponent using minimax algorithm with alpha-beta pruning
- Higher skill ceiling but slower learning

### Self-Play

- Agent plays against itself, learning from both sides
- Discovers advanced strategies through exploration
- Develops counter-strategies to its own tactics
- Most effective for developing strong play

## Training Tips

- Start with deep Q-learning and self-play for the strongest agent
- Use hyperparameter tuning to find optimal learning settings
- Train for at least 50,000 episodes for good performance
- For best results against skilled humans, train with a combination of all opponents
- Monitor win rate against MiniMax as a benchmark for strength

## Project Structure

- `main.py`: Entry point and command-line interface
- `game.py`: Core game logic for Tic Tac Toe
- `rl_agent.py`: Implementation of Q-learning and Deep Q-learning agents
- `gui.py`: Pygame-based graphical user interface
- `self_play.py`: Script for self-play training
- `hyperparameter_tuning.py`: Script for finding optimal hyperparameters

## Performance Metrics

The project includes tools to visualize:

- Win rates over time
- Game length distribution
- Reward accumulation
- Training loss (for Deep Q-learning)
