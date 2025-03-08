#!/usr/bin/env python3
import itertools
import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
from game import TicTacToe
from rl_agent import DeepQAgent, MiniMaxAgent, RandomAgent, train_agent

# Define hyperparameter search space
hyperparameters = {
    'learning_rate': [0.0005, 0.001, 0.002, 0.005],
    'discount_factor': [0.9, 0.95, 0.99],
    'exploration_rate': [0.1, 0.2, 0.3, 0.5],
    'batch_size': [32, 64, 128]
}

# Number of episodes to train for evaluation
EVAL_EPISODES = 5000

def evaluate_agent(params):
    """Train and evaluate an agent with given hyperparameters"""
    learning_rate, discount_factor, exploration_rate, batch_size = params
    
    # Create environment and agents
    env = TicTacToe()
    agent = DeepQAgent(
        player=1,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        batch_size=batch_size
    )
    
    # We'll evaluate against both random and minimax opponents
    random_opponent = RandomAgent(player=-1)
    minimax_opponent = MiniMaxAgent(player=-1)
    
    # Train against random opponent
    print(f"Training with params: lr={learning_rate}, gamma={discount_factor}, "
          f"epsilon={exploration_rate}, batch_size={batch_size}")
    agent, _ = train_agent(env, agent, random_opponent, num_episodes=EVAL_EPISODES)
    
    # Now evaluate against both opponents
    results = {}
    
    # Test against random opponent
    wins, losses, draws = evaluate_performance(agent, random_opponent, 200)
    random_win_rate = wins / (wins + losses + draws)
    results['random_win_rate'] = random_win_rate
    
    # Test against minimax opponent
    wins, losses, draws = evaluate_performance(agent, minimax_opponent, 200)
    minimax_win_rate = wins / (wins + losses + draws)
    results['minimax_win_rate'] = minimax_win_rate
    
    # Combined score (weighted more toward minimax performance)
    combined_score = 0.2 * random_win_rate + 0.8 * minimax_win_rate
    results['combined_score'] = combined_score
    
    # Save this agent with the score in filename
    agent.save(f"tuned_agent_score_{combined_score:.2f}")
    
    print(f"Random win rate: {random_win_rate:.2f}, Minimax win rate: {minimax_win_rate:.2f}, "
          f"Combined: {combined_score:.2f}")
    
    return (params, results)

def evaluate_performance(agent, opponent, num_games=100):
    """Evaluate agent performance against an opponent"""
    env = TicTacToe()
    wins, losses, draws = 0, 0, 0
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        # Randomly decide who goes first
        agent_goes_first = np.random.random() < 0.5
        
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
            
    return wins, losses, draws

def run_grid_search():
    """Run a grid search over hyperparameters"""
    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(
        hyperparameters['learning_rate'],
        hyperparameters['discount_factor'],
        hyperparameters['exploration_rate'],
        hyperparameters['batch_size']
    ))
    
    print(f"Running grid search with {len(param_combinations)} combinations")
    
    # Use multiprocessing to parallelize training
    with multiprocessing.Pool(processes=max(1, os.cpu_count() - 1)) as pool:
        results = pool.map(evaluate_agent, param_combinations)
    
    # Sort results by combined score
    results.sort(key=lambda x: x[1]['combined_score'], reverse=True)
    
    print("\nTop 5 Hyperparameter Combinations:")
    for i in range(min(5, len(results))):
        params, scores = results[i]
        lr, gamma, epsilon, batch_size = params
        print(f"{i+1}. Combined Score: {scores['combined_score']:.4f}, "
              f"Random Win Rate: {scores['random_win_rate']:.4f}, "
              f"Minimax Win Rate: {scores['minimax_win_rate']:.4f}")
        print(f"   Params: lr={lr}, gamma={gamma}, epsilon={epsilon}, batch_size={batch_size}")
    
    # Plot results
    plot_results(results)
    
    # Return best parameters
    return results[0][0]

def plot_results(results):
    """Plot hyperparameter tuning results"""
    # Extract data
    learning_rates = [r[0][0] for r in results]
    discount_factors = [r[0][1] for r in results]
    exploration_rates = [r[0][2] for r in results]
    batch_sizes = [r[0][3] for r in results]
    combined_scores = [r[1]['combined_score'] for r in results]
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot learning rate vs score
    axs[0, 0].scatter(learning_rates, combined_scores)
    axs[0, 0].set_xlabel('Learning Rate')
    axs[0, 0].set_ylabel('Combined Score')
    axs[0, 0].set_title('Learning Rate vs. Performance')
    
    # Plot discount factor vs score
    axs[0, 1].scatter(discount_factors, combined_scores)
    axs[0, 1].set_xlabel('Discount Factor')
    axs[0, 1].set_ylabel('Combined Score')
    axs[0, 1].set_title('Discount Factor vs. Performance')
    
    # Plot exploration rate vs score
    axs[1, 0].scatter(exploration_rates, combined_scores)
    axs[1, 0].set_xlabel('Exploration Rate')
    axs[1, 0].set_ylabel('Combined Score')
    axs[1, 0].set_title('Exploration Rate vs. Performance')
    
    # Plot batch size vs score
    axs[1, 1].scatter(batch_sizes, combined_scores)
    axs[1, 1].set_xlabel('Batch Size')
    axs[1, 1].set_ylabel('Combined Score')
    axs[1, 1].set_title('Batch Size vs. Performance')
    
    plt.tight_layout()
    plt.savefig('hyperparameter_tuning_results.png')

if __name__ == "__main__":
    best_params = run_grid_search()
    print("\nBest hyperparameters found:")
    print(f"Learning rate: {best_params[0]}")
    print(f"Discount factor: {best_params[1]}")
    print(f"Exploration rate: {best_params[2]}")
    print(f"Batch size: {best_params[3]}")