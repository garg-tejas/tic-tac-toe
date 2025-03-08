#!/usr/bin/env python3
import itertools
import multiprocessing
import os
import time
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
    
    process_id = multiprocessing.current_process().name
    print(f"[TUNING-{process_id}] Starting evaluation of hyperparameters")
    start_time = time.time()
    
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
    print(f"[TUNING-{process_id}] Training with params: lr={learning_rate}, gamma={discount_factor}, "
          f"epsilon={exploration_rate}, batch_size={batch_size}")
    training_start = time.time()
    agent, metrics = train_agent(env, agent, random_opponent, num_episodes=EVAL_EPISODES)
    training_time = time.time() - training_start
    print(f"[TUNING-{process_id}] Training completed in {training_time:.2f} seconds")
    
    # Now evaluate against both opponents
    results = {}
    
    # Test against random opponent
    print(f"[TUNING-{process_id}] Evaluating against random opponent...")
    eval_start = time.time()
    wins, losses, draws = evaluate_performance(agent, random_opponent, 200)
    random_win_rate = wins / (wins + losses + draws)
    results['random_win_rate'] = random_win_rate
    random_eval_time = time.time() - eval_start
    
    # Test against minimax opponent
    print(f"[TUNING-{process_id}] Evaluating against minimax opponent...")
    eval_start = time.time()
    wins, losses, draws = evaluate_performance(agent, minimax_opponent, 200)
    minimax_win_rate = wins / (wins + losses + draws)
    results['minimax_win_rate'] = minimax_win_rate
    minimax_eval_time = time.time() - eval_start
    
    # Combined score (weighted more toward minimax performance)
    combined_score = 0.2 * random_win_rate + 0.8 * minimax_win_rate
    results['combined_score'] = combined_score
    
    # Additional metrics
    results['final_exploration_rate'] = agent.exploration_rate
    results['training_time'] = training_time
    results['final_win_rate'] = metrics['win_rates'][-1] if metrics['win_rates'] else 0
    
    # Save this agent with the score in filename
    model_path = f"tuned_agent_score_{combined_score:.2f}"
    agent.save(model_path)
    
    total_time = time.time() - start_time
    print(f"[TUNING-{process_id}] Evaluation completed in {total_time:.2f} seconds")
    print(f"[TUNING-{process_id}] Results: Random win rate: {random_win_rate:.2f}, Minimax win rate: {minimax_win_rate:.2f}, "
          f"Combined: {combined_score:.2f}")
    print(f"[TUNING-{process_id}] Model saved to: {model_path}")
    
    return (params, results)

def evaluate_performance(agent, opponent, num_games=100):
    """Evaluate agent performance against an opponent"""
    process_id = multiprocessing.current_process().name
    opponent_name = opponent.__class__.__name__
    
    env = TicTacToe()
    wins, losses, draws = 0, 0, 0
    total_moves = 0
    
    # Only log progress for larger evaluations
    log_progress = num_games > 50
    progress_interval = max(1, num_games // 10)
    
    start_time = time.time()
    for game_num in range(num_games):
        state = env.reset()
        done = False
        moves = 0
        
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
            moves += 1
            
            # Switch player
            current_player = opponent if current_player == agent else agent
            
        # Determine outcome
        if env.winner == agent.player:
            wins += 1
            total_moves += moves
        elif env.winner == opponent.player:
            losses += 1
            total_moves += moves
        else:
            draws += 1
            total_moves += moves
        
        # Log progress
        if log_progress and (game_num + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            win_rate = wins / (game_num + 1)
            print(f"[TUNING-{process_id}] {opponent_name} eval: {game_num+1}/{num_games} games, "
                  f"Win rate: {win_rate:.2f}, Time: {elapsed:.2f}s")
    
    eval_time = time.time() - start_time
    avg_moves = total_moves / num_games if num_games > 0 else 0
    win_rate = wins / num_games if num_games > 0 else 0
    
    print(f"[TUNING-{process_id}] {opponent_name} evaluation complete:")
    print(f"[TUNING-{process_id}] - Win rate: {win_rate:.2f} ({wins}/{num_games})")
    print(f"[TUNING-{process_id}] - Average game length: {avg_moves:.1f} moves")
    print(f"[TUNING-{process_id}] - Evaluation time: {eval_time:.2f}s ({eval_time/num_games:.4f}s per game)")
            
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
    
    print(f"[TUNING] Starting hyperparameter tuning")
    print(f"[TUNING] Search space: {len(param_combinations)} combinations")
    print(f"[TUNING] Training episodes per configuration: {EVAL_EPISODES}")
    print(f"[TUNING] CPU cores: {os.cpu_count()}, using {max(1, os.cpu_count() - 1)} processes")
    
    for i, params in enumerate(param_combinations):
        lr, gamma, epsilon, batch_size = params
        print(f"[TUNING] Config {i+1}: lr={lr}, gamma={gamma}, epsilon={epsilon}, batch_size={batch_size}")
    
    start_time = time.time()
    
    # Use multiprocessing to parallelize training
    with multiprocessing.Pool(processes=max(1, os.cpu_count() - 1)) as pool:
        results = pool.map(evaluate_agent, param_combinations)
    
    # Sort results by combined score
    results.sort(key=lambda x: x[1]['combined_score'], reverse=True)
    
    total_time = time.time() - start_time
    avg_time = total_time / len(param_combinations) if param_combinations else 0
    
    print(f"[TUNING] Grid search completed in {total_time:.2f} seconds")
    print(f"[TUNING] Average time per configuration: {avg_time:.2f} seconds")
    
    print("\n[TUNING] Top 5 Hyperparameter Combinations:")
    for i in range(min(5, len(results))):
        params, scores = results[i]
        lr, gamma, epsilon, batch_size = params
        print(f"[TUNING] {i+1}. Combined Score: {scores['combined_score']:.4f}, "
              f"Random Win Rate: {scores['random_win_rate']:.4f}, "
              f"Minimax Win Rate: {scores['minimax_win_rate']:.4f}")
        print(f"[TUNING]    Params: lr={lr}, gamma={gamma}, epsilon={epsilon}, batch_size={batch_size}")
        print(f"[TUNING]    Training time: {scores.get('training_time', 0):.2f}s, "
              f"Final exploration: {scores.get('final_exploration_rate', 0):.4f}")
    
    # Plot results
    print("[TUNING] Generating visualization plots...")
    plot_results(results)
    
    # Return best parameters
    best_params = results[0][0]
    lr, gamma, epsilon, batch_size = best_params
    print(f"[TUNING] Best configuration found: lr={lr}, gamma={gamma}, epsilon={epsilon}, batch_size={batch_size}")
    
    return best_params

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
    print("[TUNING] Starting hyperparameter tuning process")
    start_time = time.time()
    
    best_params = run_grid_search()
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print("\n[TUNING] Hyperparameter tuning completed")
    print(f"[TUNING] Total time: {hours}h {minutes}m {seconds}s")
    print(f"[TUNING] Best hyperparameters found:")
    print(f"[TUNING] - Learning rate: {best_params[0]}")
    print(f"[TUNING] - Discount factor: {best_params[1]}")
    print(f"[TUNING] - Exploration rate: {best_params[2]}")
    print(f"[TUNING] - Batch size: {best_params[3]}")
    print(f"[TUNING] Hyperparameter tuning plots saved to: hyperparameter_tuning_results.png")