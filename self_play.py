#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from game import TicTacToe
from rl_agent import DeepQAgent, QAgent, MiniMaxAgent, RandomAgent, train_agent # train_agent is now parallelized

def self_play_training(agent_type='deep', num_episodes=50000, save_path=None, visualize=True, num_processes=8): # Add num_processes here
    """Train an agent through self-play"""
    print("[SELF-PLAY] Starting self-play training process")
    env = TicTacToe()

    # Create agent
    print(f"[SELF-PLAY] Initializing {agent_type} agent")
    if agent_type == 'deep':
        agent = DeepQAgent(player=1,
                          exploration_rate=0.3,
                          learning_rate=0.001,
                          discount_factor=0.99,
                          batch_size=64,
                          memory_size=20000)
        print(f"[SELF-PLAY] Created DeepQAgent with exploration={agent.exploration_rate:.2f}, learning_rate={agent.learning_rate:.4f}")
        print(f"[SELF-PLAY] Neural network parameters: batch_size={agent.batch_size}, memory_size={agent.memory_size}")
    else:
        agent = QAgent(player=1,
                      exploration_rate=0.3,
                      learning_rate=0.2,
                      discount_factor=0.95)
        print(f"[SELF-PLAY] Created QAgent with exploration={agent.exploration_rate:.2f}, learning_rate={agent.learning_rate:.2f}")

    # Use None for opponent in self-play mode
    print(f"[SELF-PLAY] Starting training for {num_episodes} episodes using {num_processes} processes...") # Log processes
    start_time = time.time()
    agent, metrics = train_agent(env, agent, None, num_episodes=num_episodes, self_play=True, num_processes=num_processes) # Pass num_processes
    training_time = time.time() - start_time
    print(f"[SELF-PLAY] Training completed in {training_time:.2f} seconds ({training_time/num_episodes:.4f} sec/episode avg across processes)") # Updated log

    # Save the agent
    if save_path is None:
        save_path = f"{agent_type}_self_play_agent"

    agent.save(save_path)
    print(f"[SELF-PLAY] Agent saved to {save_path}")

    # Test against minimax to gauge strength
    print("[SELF-PLAY] Evaluating agent against minimax opponent...")
    wins, losses, draws = test_agent_performance(agent, 'minimax', 100)
    print(f"[SELF-PLAY] Performance summary - Win rate: {wins/100:.2f}, Loss rate: {losses/100:.2f}, Draw rate: {draws/100:.2f}")

    # Visualize training metrics
    if visualize:
        print("[SELF-PLAY] Generating training metrics visualization...")
        plot_metrics(metrics)

    return agent, metrics

def plot_metrics(metrics):
    """Plot training metrics"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Plot win rates
    win_rates = metrics['win_rates']
    x_vals = np.arange(len(win_rates)) * args.processes * 10 # Adjusted x-axis for batch updates
    ax1.plot(x_vals, win_rates)
    ax1.set_title('Win Rate During Self-Play Training')
    ax1.set_xlabel('Episode (Batch of {})'.format(args.processes)) # Updated label
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

    print(f"[EVAL] Testing agent against {opponent_type} opponent ({num_games} games)")

    if opponent_type == 'random':
        opponent = RandomAgent(player=-1)
    elif opponent_type == 'minimax':
        opponent = MiniMaxAgent(player=-1)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    wins, losses, draws = 0, 0, 0
    total_moves = 0

    start_time = time.time()
    for i in range(num_games):
        state = env.reset()
        done = False
        moves = 0

        # Alternate who goes first
        agent_goes_first = i % 2 == 0

        if agent_goes_first:
            current_player = agent
            print(f"[EVAL] Game {i+1}/{num_games}: Agent goes first")
        else:
            current_player = opponent
            print(f"[EVAL] Game {i+1}/{num_games}: Opponent goes first")

        # Play a game
        while not done:
            if current_player == agent:
                action = agent.choose_action(state, training=False)
            else:
                action = opponent.choose_action(state)

            if action is None:
                print(f"[EVAL] Game {i+1}: No valid moves available")
                break

            state, reward, done = env.make_move(action)
            moves += 1

            # Switch player
            current_player = opponent if current_player == agent else agent

        # Determine outcome
        if env.winner == agent.player:
            wins += 1
            result = "WIN"
        elif env.winner == opponent.player:
            losses += 1
            result = "LOSS"
        else:
            draws += 1
            result = "DRAW"

        total_moves += moves
        print(f"[EVAL] Game {i+1} result: {result} in {moves} moves")

        # Print progress every 10 games
        if (i+1) % 10 == 0:
            progress_time = time.time() - start_time
            print(f"[EVAL] Progress: {i+1}/{num_games} games, {progress_time:.1f} seconds")
            print(f"[EVAL] Current stats - Wins: {wins}, Losses: {losses}, Draws: {draws}")

    eval_time = time.time() - start_time
    avg_moves = total_moves / num_games

    print(f"[EVAL] Evaluation complete in {eval_time:.2f} seconds ({eval_time/num_games:.2f} sec/game)")
    print(f"[EVAL] Performance against {opponent_type}:")
    print(f"[EVAL] - Wins: {wins}/{num_games} ({wins/num_games:.2f})")
    print(f"[EVAL] - Losses: {losses}/{num_games} ({losses/num_games:.2f})")
    print(f"[EVAL] - Draws: {draws}/{num_games} ({draws/num_games:.2f})")
    print(f"[EVAL] - Average game length: {avg_moves:.1f} moves")

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
    parser.add_argument('--processes', type=int, default=8, help='Number of parallel processes for training') # Added processes argument
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args() # Parse arguments here
    agent, metrics = self_play_training(
        agent_type=args.agent,
        num_episodes=args.episodes,
        save_path=args.save_path,
        visualize=not args.no_visualize,
        num_processes=args.processes # Pass num_processes here
    )