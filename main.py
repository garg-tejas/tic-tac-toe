#!/usr/bin/env python3
import sys
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from game import TicTacToe
from rl_agent import QAgent, RandomAgent, MiniMaxAgent, DeepQAgent, train_agent # train_agent is now parallelized version
from gui import TicTacToeGUI

def parse_args():
    parser = argparse.ArgumentParser(description='Tic Tac Toe with Reinforcement Learning')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--agent', type=str, default='deep', choices=['q', 'deep'], help='Agent type: q or deep')
    parser.add_argument('--episodes', type=int, default=50000, help='Number of training episodes')
    parser.add_argument('--opponent', type=str, default='random', choices=['random', 'minimax', 'self'],
                        help='Opponent type: random, minimax, or self-play')
    parser.add_argument('--play', action='store_true', help='Play against the trained agent')
    parser.add_argument('--visualize', action='store_true', help='Visualize training metrics')
    parser.add_argument('--save-path', type=str, default=None, help='Path to save the trained agent')
    parser.add_argument('--load-path', type=str, default=None, help='Path to load the agent from')
    parser.add_argument('--processes', type=int, default=8, help='Number of parallel processes for training') # Added processes argument
    return parser.parse_args()

def create_agent(agent_type, player=-1, load_path=None):
    if agent_type == 'q':
        agent = QAgent(player=player)
    else:  # deep
        agent = DeepQAgent(player=player)

    if load_path:
        agent.load(load_path)
        print(f"Agent loaded from {load_path}")

    return agent

def create_opponent(opponent_type, player=1):
    if opponent_type == 'random':
        return RandomAgent(player=player)
    elif opponent_type == 'minimax':
        return MiniMaxAgent(player=player)
    else:
        return None  # For self-play

def visualize_metrics(metrics):
    """Visualize training metrics"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Plot win rates
    win_rates = metrics['win_rates']
    x_vals = np.arange(len(win_rates)) * args.processes * 10 # Adjusted x-axis for batch updates
    ax1.plot(x_vals, win_rates)
    ax1.set_title('Win Rate During Training')
    ax1.set_xlabel('Episodes (Batch of {})'.format(args.processes)) # Updated label
    ax1.set_ylabel('Win Rate')

    # Plot rewards
    rewards = metrics['rewards']
    window_size = min(100, len(rewards))
    if window_size > 0:
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(smoothed_rewards)
        ax2.set_title('Smoothed Rewards')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')

    # Plot episode lengths
    lengths = metrics['episode_lengths']
    window_size = min(100, len(lengths))
    if window_size > 0:
        smoothed_lengths = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
        ax3.plot(smoothed_lengths)
        ax3.set_title('Average Game Length')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Moves')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

def train_and_save(agent_type, opponent_type, num_episodes, save_path=None, num_processes=8): # Added num_processes here
    """Train the agent and save the model"""
    print("[MAIN] Starting training session")
    env = TicTacToe()

    # Create agent
    agent = create_agent(agent_type)
    print(f"[MAIN] Created {agent_type} agent with exploration rate {agent.exploration_rate:.4f}")

    # Create opponent or use self-play
    self_play = opponent_type == 'self'
    if self_play:
        opponent = None
        print("[MAIN] Using self-play training mode")
    else:
        opponent = create_opponent(opponent_type)
        print(f"[MAIN] Using {opponent_type} opponent for training")

    # Train the agent
    print(f"[MAIN] Training {agent_type} agent against {opponent_type} for {num_episodes} episodes using {num_processes} processes...") # Log processes
    start_time = time.time()
    agent, metrics = train_agent(env, agent, opponent, num_episodes=num_episodes, num_processes=num_processes, self_play=self_play) # Pass num_processes
    training_time = time.time() - start_time
    print(f"[MAIN] Parallel Training completed in {training_time:.2f} seconds ({training_time/num_episodes:.4f} sec/episode avg across processes)") # Updated log

    # Print final training stats
    print(f"[MAIN] Final training stats:")
    print(f"[MAIN] - Win rate: {metrics['win_rates'][-1]:.4f}")
    print(f"[MAIN] - Average reward: {np.mean(metrics['rewards'][-1000:]):.4f}")
    print(f"[MAIN] - Final exploration rate: {agent.exploration_rate:.4f}")

    # Save the agent
    if save_path is None:
        if agent_type == 'q':
            save_path = 'q_agent'
        else:
            save_path = 'deep_q_agent'

    agent.save(save_path)
    print(f"[MAIN] Agent saved to {save_path}")

    return agent, metrics

def main():
    global args # Make args global if you intend to modify it or access it globally in visualize_metrics
    args = parse_args()

    if args.train:
        agent, metrics = train_and_save(args.agent, args.opponent, args.episodes, args.save_path, num_processes=args.processes) # Pass num_processes

        if args.visualize:
            visualize_metrics(metrics)

    if args.play or not args.train:
        # Load the trained agent if not just trained
        if not args.train:
            load_path = args.load_path or ('q_agent' if args.agent == 'q' else 'deep_q_agent')
            agent = create_agent(args.agent, load_path=load_path)

        # Start the GUI application
        gui = TicTacToeGUI(agent=agent)
        gui.run()

if __name__ == "__main__":
    main()