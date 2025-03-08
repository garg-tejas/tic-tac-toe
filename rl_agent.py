import numpy as np
import pickle
import os
import random
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class QAgent:
    def __init__(self, player=-1, exploration_rate=0.2, learning_rate=0.2, discount_factor=0.9):
        self.player = player  # -1 for O, 1 for X
        self.q_table = defaultdict(lambda: np.zeros(9))
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.training_history = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'rewards': []
        }
        self.last_state = None
        self.last_action = None
        
    def board_to_state(self, board):
        """Convert board to a hashable state representation for the Q-table"""
        # Flatten and convert to tuple for hashability
        return tuple(board.flatten())
    
    def action_to_position(self, action):
        """Convert flat action index to board position"""
        return action // 3, action % 3
    
    def position_to_action(self, position):
        """Convert board position to flat action index"""
        i, j = position
        return i * 3 + j
    
    def get_valid_actions(self, board):
        """Get list of valid actions from board state"""
        valid_actions = []
        for i in range(3):
            for j in range(3):
                if board[i, j] == 0:
                    valid_actions.append(self.position_to_action((i, j)))
        return valid_actions
    
    def choose_action(self, board, training=True, look_ahead=True):
        """Choose an action using epsilon-greedy policy with optional look-ahead"""
        state = self.board_to_state(board)
        valid_actions = self.get_valid_actions(board)
        
        if not valid_actions:
            return None
        
        # Explore: random action
        if training and np.random.random() < self.exploration_rate:
            action = random.choice(valid_actions)
        # Exploit: best known action
        else:
            # Filter Q-values for valid actions only
            q_values = self.q_table[state]
            masked_q_values = np.full(9, -np.inf)
            for action in valid_actions:
                masked_q_values[action] = q_values[action]
            
            # If look-ahead is enabled and not in training mode, evaluate next states
            if look_ahead and not training:
                # Look ahead to find potential winning moves or block opponent wins
                for action in valid_actions:
                    i, j = self.action_to_position(action)
                    # Create a copy of the board to simulate the move
                    board_copy = board.copy()
                    board_copy[i, j] = self.player
                    
                    # If this move leads to a win, choose it immediately
                    if self._check_win(board_copy, self.player):
                        return (i, j)
                    
                    # Check if opponent has a winning move to block
                    for opp_action in valid_actions:
                        if opp_action == self.position_to_action((i, j)):
                            continue  # Skip the spot we just filled
                            
                        opp_i, opp_j = self.action_to_position(opp_action)
                        board_copy2 = board.copy()
                        board_copy2[opp_i, opp_j] = -self.player  # Opponent's move
                        
                        if self._check_win(board_copy2, -self.player):
                            # Opponent would win here, block it
                            masked_q_values[opp_action] += 5.0  # Boost blocking moves
                
                # Give corner and center positions a slight advantage
                strategic_positions = [0, 2, 4, 6, 8]  # corners and center
                for pos in strategic_positions:
                    if pos in valid_actions:
                        masked_q_values[pos] += 0.2  # Small boost for strategic positions
                
                # If center is open in the beginning, prioritize it
                if len(valid_actions) > 7 and 4 in valid_actions:  # Beginning of game
                    masked_q_values[4] += 0.5  # Boost center position at start
            
            # Choose the best action (with random tie-breaking)
            max_q = np.max(masked_q_values)
            best_actions = [a for a in valid_actions if masked_q_values[a] == max_q]
            action = random.choice(best_actions)
        
        return self.action_to_position(action)
    
    def _check_win(self, board, player):
        """Check if player has won on the given board"""
        # Check rows, columns, and diagonals
        for i in range(3):
            # Check rows
            if np.sum(board[i, :]) == player * 3:
                return True
            # Check columns
            if np.sum(board[:, i]) == player * 3:
                return True
        
        # Check diagonals
        if np.sum(np.diag(board)) == player * 3:
            return True
        if np.sum(np.diag(np.fliplr(board))) == player * 3:
            return True
        
        return False
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning algorithm"""
        state_key = self.board_to_state(state)
        
        # Skip update if action is None (this happens when learning from opponent winning)
        if action is None:
            return
            
        action_idx = self.position_to_action(action)
        
        if done:
            # Terminal state
            target = reward
        else:
            next_state_key = self.board_to_state(next_state)
            # Get best Q-value for next state
            next_q_value = np.max(self.q_table[next_state_key])
            target = reward + self.discount_factor * next_q_value
        
        # Update Q-value for current state-action pair
        old_value = self.q_table[state_key][action_idx]
        self.q_table[state_key][action_idx] += self.learning_rate * (target - old_value)
    
    def update_training_history(self, reward):
        """Update training statistics"""
        self.training_history['rewards'].append(reward)
        if reward == 1:  # Win
            self.training_history['wins'] += 1
        elif reward == -1:  # Loss
            self.training_history['losses'] += 1
        elif reward == 0.5:  # Draw
            self.training_history['draws'] += 1
    
    def save(self, filepath='q_agent.pkl'):
        """Save the Q-table and training history to a file"""
        # Convert defaultdict to regular dict for saving
        data = {
            'q_table': dict(self.q_table),
            'training_history': self.training_history,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'player': self.player
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath='q_agent.pkl'):
        """Load the Q-table and training history from a file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            # Convert dict back to defaultdict
            q_table = defaultdict(lambda: np.zeros(9))
            q_table.update(data['q_table'])
            self.q_table = q_table
            
            self.training_history = data['training_history']
            self.exploration_rate = data['exploration_rate']
            self.learning_rate = data['learning_rate']
            self.discount_factor = data['discount_factor']
            self.player = data['player']
            return True
        return False
    
    def decrease_exploration(self, factor=0.995, min_rate=0.01):
        """Decrease exploration rate over time"""
        self.exploration_rate = max(self.exploration_rate * factor, min_rate)
        
    def get_win_rate(self):
        """Calculate win rate from training history"""
        total_games = self.training_history['wins'] + self.training_history['losses'] + self.training_history['draws']
        if total_games == 0:
            return 0
        return self.training_history['wins'] / total_games


class RandomAgent:
    """Simple random agent for training"""
    def __init__(self, player=1):
        self.player = player
    
    def choose_action(self, board, training=False):
        valid_moves = [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]
        if not valid_moves:
            return None
        return random.choice(valid_moves)
    
    def learn(self, state, action, reward, next_state, done):
        pass  # Random agent doesn't learn


class MiniMaxAgent:
    """Advanced opponent using minimax with alpha-beta pruning"""
    def __init__(self, player=1, depth=9):
        self.player = player
        self.max_depth = depth
    
    def choose_action(self, board, training=False):
        valid_moves = [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]
        if not valid_moves:
            return None
        
        best_score = -float('inf')
        best_move = valid_moves[0]
        
        for move in valid_moves:
            # Make tentative move
            new_board = board.copy()
            i, j = move
            new_board[i, j] = self.player
            
            # Evaluate move
            score = self._minimax(new_board, 0, False, -float('inf'), float('inf'), -self.player)
            
            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _minimax(self, board, depth, is_maximizing, alpha, beta, current_player):
        # Check for terminal states
        winner = self._check_winner(board)
        if winner is not None:
            return 10 if winner == self.player else -10
        
        if depth >= self.max_depth or self._is_board_full(board):
            return 0
        
        if is_maximizing:
            best_score = -float('inf')
            for i in range(3):
                for j in range(3):
                    if board[i, j] == 0:
                        board[i, j] = current_player
                        score = self._minimax(board, depth + 1, False, alpha, beta, -current_player)
                        board[i, j] = 0  # Undo move
                        best_score = max(score, best_score)
                        alpha = max(alpha, best_score)
                        if beta <= alpha:
                            break
            return best_score
        else:
            best_score = float('inf')
            for i in range(3):
                for j in range(3):
                    if board[i, j] == 0:
                        board[i, j] = current_player
                        score = self._minimax(board, depth + 1, True, alpha, beta, -current_player)
                        board[i, j] = 0  # Undo move
                        best_score = min(score, best_score)
                        beta = min(beta, best_score)
                        if beta <= alpha:
                            break
            return best_score
    
    def _check_winner(self, board):
        # Check rows, columns, and diagonals
        for i in range(3):
            # Check rows
            if abs(np.sum(board[i, :])) == 3:
                return board[i, 0]
            # Check columns
            if abs(np.sum(board[:, i])) == 3:
                return board[0, i]
        
        # Check diagonals
        if abs(np.sum(np.diag(board))) == 3:
            return board[0, 0]
        if abs(np.sum(np.diag(np.fliplr(board)))) == 3:
            return board[0, 2]
        
        return None
    
    def _is_board_full(self, board):
        return not any(board[i, j] == 0 for i in range(3) for j in range(3))
    
    def learn(self, state, action, reward, next_state, done):
        pass  # MiniMax agent doesn't learn


class DeepQAgent:
    def __init__(self, player=-1, exploration_rate=0.3, learning_rate=0.001, discount_factor=0.95, batch_size=64, memory_size=10000):
        self.player = player  # -1 for O, 1 for X
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = []  # Experience replay buffer
        self.model = self._build_model()
        self.target_model = self._build_model()  # Target network for more stable learning
        self.update_target_model()  # Sync weights
        self.training_history = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'rewards': [],
            'loss': []
        }
        self.last_state = None
        self.last_action = None
        
    def _build_model(self):
        """Build a neural network model for deep Q-learning"""
        model = Sequential()
        # Input layer represents the flattened game board (9 positions)
        model.add(Input(shape=(18,)))  # 9 positions * 2 (one-hot encoding for X and O)
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(9, activation='linear'))  # Output layer - Q-values for each position
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update target model with weights from main model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def state_to_input(self, board):
        """Convert board state to neural network input"""
        # One-hot encoding for X and O
        input_state = np.zeros((18,))
        flat_board = board.flatten()
        
        for i in range(9):
            if flat_board[i] == 1:  # X
                input_state[i] = 1
            elif flat_board[i] == -1:  # O
                input_state[i + 9] = 1
                
        return input_state.reshape(1, -1)  # Reshape for model input
    
    def remember(self, state, action, reward, next_state, done):
        """Add experience to memory buffer"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)  # Remove oldest experience
        
        # Convert action tuple to index
        if action is not None:
            action_idx = action[0] * 3 + action[1]
            self.memory.append((state, action_idx, reward, next_state, done))
    
    def replay(self, batch_size=None):
        """Train the model by replaying experiences"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return 0  # Not enough experiences
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        loss = 0
        
        for state, action_idx, reward, next_state, done in minibatch:
            state_input = self.state_to_input(state)
            next_state_input = self.state_to_input(next_state)
            
            target = self.model.predict(state_input, verbose=0)[0]
            
            if done:
                target[action_idx] = reward
            else:
                # Double DQN: Select action using online network, evaluate with target network
                next_action = np.argmax(self.model.predict(next_state_input, verbose=0)[0])
                future_reward = self.target_model.predict(next_state_input, verbose=0)[0][next_action]
                target[action_idx] = reward + self.discount_factor * future_reward
            
            # Train the model
            history = self.model.fit(state_input, np.array([target]), epochs=1, verbose=0)
            loss += history.history['loss'][0]
        
        # Record loss
        self.training_history['loss'].append(loss / batch_size)
        return loss / batch_size
    
    def choose_action(self, board, training=True, look_ahead=True):
        """Choose an action using epsilon-greedy policy with look-ahead"""
        valid_actions = self.get_valid_actions(board)
        
        if not valid_actions:
            return None
        
        # Explore: random action
        if training and np.random.random() < self.exploration_rate:
            return random.choice(valid_actions)
        
        # Exploit: best action from model
        state_input = self.state_to_input(board)
        q_values = self.model.predict(state_input, verbose=0)[0]
        
        # Mask invalid actions
        for i in range(3):
            for j in range(3):
                action_idx = i * 3 + j
                if (i, j) not in valid_actions:
                    q_values[action_idx] = -np.inf
        
        # Look ahead enhancements
        if look_ahead and not training:
            # Check for winning moves or blocks
            for i, j in valid_actions:
                action_idx = i * 3 + j
                
                # Check if this move leads to a win
                board_copy = board.copy()
                board_copy[i, j] = self.player
                if self._check_win(board_copy, self.player):
                    return (i, j)  # Immediate win
                
                # Check if opponent has a winning move to block
                opponent = -self.player
                for oi, oj in valid_actions:
                    if (oi, oj) == (i, j):
                        continue
                    
                    board_copy = board.copy()
                    board_copy[oi, oj] = opponent
                    if self._check_win(board_copy, opponent):
                        q_values[i * 3 + j] += 5.0  # Boost blocking moves
                
            # Strategic position boosts
            strategic_positions = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]  # corners and center
            for pos in strategic_positions:
                if pos in valid_actions:
                    i, j = pos
                    q_values[i * 3 + j] += 0.2
            
            # Prioritize center at beginning
            if len(valid_actions) > 7 and (1, 1) in valid_actions:
                q_values[1 * 3 + 1] += 0.5
        
        # Choose best action
        best_action_idx = np.argmax(q_values)
        return (best_action_idx // 3, best_action_idx % 3)
    
    def _check_win(self, board, player):
        """Check if player has won on the given board"""
        # Check rows, columns, and diagonals
        for i in range(3):
            # Check rows
            if np.sum(board[i, :]) == player * 3:
                return True
            # Check columns
            if np.sum(board[:, i]) == player * 3:
                return True
        
        # Check diagonals
        if np.sum(np.diag(board)) == player * 3:
            return True
        if np.sum(np.diag(np.fliplr(board))) == player * 3:
            return True
        
        return False
    
    def get_valid_actions(self, board):
        """Get list of valid actions from board state"""
        valid_actions = []
        for i in range(3):
            for j in range(3):
                if board[i, j] == 0:
                    valid_actions.append((i, j))
        return valid_actions
    
    def learn(self, state, action, reward, next_state, done):
        """Add experience to memory and train model if enough samples"""
        self.remember(state, action, reward, next_state, done)
        
        # Train model
        if len(self.memory) >= self.batch_size:
            self.replay()
            
        # Update target model periodically
        if len(self.memory) % 500 == 0:
            self.update_target_model()
    
    def update_training_history(self, reward):
        """Update training statistics"""
        self.training_history['rewards'].append(reward)
        if reward == 1:  # Win
            self.training_history['wins'] += 1
        elif reward == -1:  # Loss
            self.training_history['losses'] += 1
        elif reward == 0.5:  # Draw
            self.training_history['draws'] += 1
    
    def save(self, filepath='deep_q_agent'):
        """Save the model and training history"""
        # Save neural network model
        self.model.save(f"{filepath}_model.h5")
        
        # Save training history and parameters
        data = {
            'training_history': self.training_history,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'player': self.player,
            'batch_size': self.batch_size,
            'memory_size': self.memory_size
        }
        with open(f"{filepath}_data.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath='deep_q_agent'):
        """Load the model and training history"""
        model_path = f"{filepath}_model.h5"
        data_path = f"{filepath}_data.pkl"
        
        if os.path.exists(model_path) and os.path.exists(data_path):
            # Load neural network model
            self.model = load_model(model_path)
            self.target_model = load_model(model_path)
            
            # Load training history and parameters
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.training_history = data['training_history']
            self.exploration_rate = data['exploration_rate']
            self.learning_rate = data['learning_rate']
            self.discount_factor = data['discount_factor']
            self.player = data['player']
            self.batch_size = data.get('batch_size', 64)
            self.memory_size = data.get('memory_size', 10000)
            return True
        return False
    
    def decrease_exploration(self, factor=0.995, min_rate=0.01):
        """Decrease exploration rate over time"""
        self.exploration_rate = max(self.exploration_rate * factor, min_rate)
    
    def get_win_rate(self):
        """Calculate win rate from training history"""
        total_games = self.training_history['wins'] + self.training_history['losses'] + self.training_history['draws']
        if total_games == 0:
            return 0
        return self.training_history['wins'] / total_games
    
    def plot_training_history(self):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot rewards
        rewards = self.training_history['rewards']
        window_size = min(100, len(rewards))
        if window_size > 0:
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(smoothed_rewards)
            ax1.set_title('Smoothed Rewards')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
        
        # Plot loss
        loss = self.training_history['loss']
        if loss:
            window_size = min(100, len(loss))
            if window_size > 0:
                smoothed_loss = np.convolve(loss, np.ones(window_size)/window_size, mode='valid')
                ax2.plot(smoothed_loss)
                ax2.set_title('Training Loss')
                ax2.set_xlabel('Training Step')
                ax2.set_ylabel('Loss')
        
        plt.tight_layout()
        return fig


def train_agent(env, agent, opponent, num_episodes=10000, decay_exploration=True, self_play=False):
    """Train the RL agent against an opponent or through self-play"""
    # Create performance metrics
    win_rates = []
    rewards_history = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        moves = 0
        
        # Determine who goes first - enhanced probability for both positions
        agent_goes_first = random.random() < 0.5
        
        if agent_goes_first:
            # Agent goes first
            while not done:
                # Agent's turn
                moves += 1
                action = agent.choose_action(state, training=True)
                if action is None:
                    break
                next_state, reward, done = env.make_move(action)
                
                if done:
                    # Reward shaping: shorter games are better if winning
                    if reward > 0:
                        reward = reward * (1 + 1/moves)  # Bonus for quick wins
                    agent.learn(state, action, reward, next_state, done)
                    episode_reward += reward
                    break
                
                # Opponent's turn (or self-play)
                moves += 1
                if self_play:
                    # In self-play, the agent plays against itself
                    opp_action = agent.choose_action(next_state, training=True)
                else:
                    opp_action = opponent.choose_action(next_state)
                    
                if opp_action is None:
                    break
                next_next_state, opp_reward, done = env.make_move(opp_action)
                
                # Agent learns from opponent's action (even in self-play)
                penalty = -opp_reward * 1.5 if done and opp_reward > 0 else -opp_reward
                agent.learn(state, action, penalty, next_next_state, done)
                episode_reward += penalty
                state = next_next_state
        else:
            # Opponent goes first
            while not done:
                # Opponent's turn (or self-play agent)
                moves += 1
                if self_play:
                    opp_action = agent.choose_action(state, training=True)
                else:
                    opp_action = opponent.choose_action(state)
                    
                if opp_action is None:
                    break
                next_state, opp_reward, done = env.make_move(opp_action)
                
                if done:
                    # Harsh penalty for losing
                    if opp_reward > 0:
                        agent.learn(state, None, -2.0, next_state, done)  # Extra penalty for losing
                    break
                
                # Agent's turn
                moves += 1
                action = agent.choose_action(next_state, training=True)
                if action is None:
                    break
                next_next_state, reward, done = env.make_move(action)
                
                # Enhanced reward for winning from second position
                if done and reward > 0:
                    reward = reward * (1 + 1/moves)  # Bonus for quick wins
                    reward *= 2.0  # Extra reward for winning from second position
                
                agent.learn(next_state, action, reward, next_next_state, done)
                episode_reward += reward
                state = next_next_state
        
        # Update training history
        agent.update_training_history(episode_reward)
        rewards_history.append(episode_reward)
        episode_lengths.append(moves)
        
        # Decay exploration rate
        if decay_exploration and (episode + 1) % 100 == 0:
            agent.decrease_exploration()
        
        # Update win rate every 100 episodes
        if (episode + 1) % 100 == 0:
            win_rates.append(agent.get_win_rate())
        
        # Save progress periodically
        if (episode + 1) % 1000 == 0:
            agent.save()
            print(f"Episode {episode + 1}/{num_episodes} - Win rate: {agent.get_win_rate():.2f}, Exploration rate: {agent.exploration_rate:.4f}")
            
            # Plot progress if using Deep Q agent
            if hasattr(agent, 'plot_training_history'):
                agent.plot_training_history()
                plt.savefig(f'training_progress_episode_{episode+1}.png')
                plt.close()
    
    # Final training metrics
    metrics = {
        'win_rates': win_rates,
        'rewards': rewards_history,
        'episode_lengths': episode_lengths
    }
    
    return agent, metrics