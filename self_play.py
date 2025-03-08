#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from game import TicTacToe
from rl_agent import DeepQAgent, QAgent, MiniMaxAgent, RandomAgent, train_agent

def self_play_training(agent_type='deep', num_episodes=50000, save_path=None, visualize=True):
    """Train an agent through self-play"""
    env = TicTacToe()
    
    # Create agent
    if agent_type == 'deep':
        agent = DeepQAgent(player=1, 
                          exploration_rate=0.3, 
                          learning_rate=0.001, 
                          discount_factor=0.99,
                          batch_size=64,
                          memory_size=20000)
    else:
        agent = QAgent(player=1,
                      exploration_rate=0.3,
                      learning_rate=0.2,
                      discount_factor=0.95)
    
    # Use None for opponent in self-play mode
    print(f"Training {agent_type} agent with self-play for {num_episodes} episodes...")
    agent, metrics = train_agent(env, agent, None, num_episodes=num_episodes, self_play=True)
    
    # Save the agent
    if save_path is None:
        save_path = f"{agent_type}_self_play_agent"
    
    agent.save(save_path)
    print(f"Agent saved to {save_path}")
    
    # Test against minimax to gauge strength
    test_agent_performance(agent, 'minimax', 100)
    
    # Visualize training metrics
    if visualize:
        plot_metrics(metrics)
    
    return agent, metrics

def plot_metrics(metrics):
    """Plot training metrics"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot win rates
    win_rates = metrics['win_rates']
    x_vals = np.arange(len(win_rates)) * 100
    ax1.plot(x_vals, win_rates)
    ax1.set_title('Win Rate During Self-Play Training')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Win Rate')
    
    # Plot rewards
    rewards = metrics['rewards']
    window_size = min(500, len(rewards))
    if window_size > 0:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(smoothed_rewards)
        ax2.set_title('Smoothed Rewards')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
    
    # Plot episode lengths
    lengths = metrics['episode_lengths']
    window_size = min(500, len(lengths))
    if window_size > 0:
        smoothed_lengths = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
        ax3.plot(smoothed_lengths)
        ax3.set_title('Average Game Length')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Moves')
    
    plt.tight_layout()
    plt.savefig('self_play_training_metrics.png')
    plt.show()

def test_agent_performance(agent, opponent_type, num_games=100):
    """Test agent performance against a specific opponent"""
    env = TicTacToe()
    
    if opponent_type == 'random':
        opponent = RandomAgent(player=-1)
    elif opponent_type == 'minimax':
        opponent = MiniMaxAgent(player=-1)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    wins, losses, draws = 0, 0, 0
    
    for i in range(num_games):
        state = env.reset()
        done = False
        
        # Alternate who goes first
        agent_goes_first = i % 2 == 0
        
        if agent_goes_first:
            current_player = agent
        else:
            current_player = opponent
        
        # Play a game
        while not done:
            if current_player == agent:
                action = agent.choose_action(state, training=False)
            else:
                action = opponent.choose_action(state)
                
            if action is None:
                break
                
            state, reward, done = env.make_move(action)
            
            # Switch player
            current_player = opponent if current_player == agent else agent
            
        # Determine outcome
        if env.winner == agent.player:
            wins += 1
        elif env.winner == opponent.player:
            losses += 1
        else:
            draws += 1
    
    print(f"Performance against {opponent_type}:")
    print(f"Wins: {wins}/{num_games} ({wins/num_games:.2f})")
    print(f"Losses: {losses}/{num_games} ({losses/num_games:.2f})")
    print(f"Draws: {draws}/{num_games} ({draws/num_games:.2f})")
    
    return wins, losses, draws

def parse_args():
    parser = argparse.ArgumentParser(description='Self-play training for Tic Tac Toe RL')
    parser.add_argument('--agent', type=str, default='deep', choices=['q', 'deep'], 
                        help='Agent type: q or deep')
    parser.add_argument('--episodes', type=int, default=50000, 
                        help='Number of training episodes')
    parser.add_argument('--save-path', type=str, default=None, 
                        help='Path to save the trained agent')
    parser.add_argument('--no-visualize', action='store_true', 
                        help='Disable visualization of training metrics')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    agent, metrics = self_play_training(
        agent_type=args.agent,
        num_episodes=args.episodes,
        save_path=args.save_path,
        visualize=not args.no_visualize
    )