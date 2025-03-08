# Tic Tac Toe RL

A Tic Tac Toe game with a Reinforcement Learning (RL) agent. The game features a graphical user interface where you can watch the agent train and play against it.

## Features

- Play Tic Tac Toe against a trained RL agent
- Train the agent in real-time with visual feedback
- Choose between playing as X or O
- Select different opponent types for training (Random or MiniMax)
- Save and load trained agent models
- Real-time visualization of training progress and win rates

## Requirements

- Python 3.6+
- NumPy
- Pygame

## Installation

1. Clone this repository
2. Install dependencies:
```
pip install numpy pygame
```

## Usage

Run the main script to start the game:
```
python main.py
```

### Controls

- Click on the board to place your symbol when it's your turn
- Use the "Train Agent" button to start training the RL agent
- Press ESC to stop training
- Use "Reset Game" to start a new game
- "Save Agent" and "Load Agent" buttons allow you to save and load trained models
- Use the radio buttons to select your preferred opponent type and whether you want to play as X or O

## How It Works

The game uses a Q-learning algorithm to train an agent to play Tic Tac Toe. The agent learns by playing thousands of games against different opponents and updating its strategy based on rewards received.

### RL Agent

The agent uses:
- Q-learning with epsilon-greedy exploration
- Decreasing exploration rate over time
- State representation as board configurations
- Rewards for winning, losing, and drawing

### Opponents

- Random Agent: Makes random valid moves (good for early training)
- MiniMax Agent: Uses the minimax algorithm with alpha-beta pruning (challenging opponent)

## Training Tips

- Start training against the Random opponent to learn basic strategies
- After achieving a good win rate against the Random opponent, train against the MiniMax opponent
- Training for at least 5000 episodes is recommended for good performance
- The agent learns faster when the exploration rate is higher, but performs better in actual games with a lower exploration rate

## Project Structure

- `main.py`: Entry point for the application
- `game.py`: Core game logic for Tic Tac Toe
- `rl_agent.py`: Implementation of the Q-learning agent and opponents
- `gui.py`: Pygame-based graphical user interface

## License

MIT