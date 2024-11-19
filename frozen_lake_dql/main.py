# frozen_lake_dql/main.py
"""Main script for training and evaluating DQN agent on FrozenLake environment."""
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from frozen_lake_dql.training.trainer import FrozenLakeTrainer
from frozen_lake_dql.utils.config import DQNConfig, TrainingConfig
from frozen_lake_dql.utils.visualization import (
    plot_training_metrics,
    plot_q_values,
    plot_episode_visualization
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DQN agent on FrozenLake')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                      help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=100,
                      help='Maximum steps per episode')
    parser.add_argument('--slippery', action='store_true',
                      help='Make environment slippery')
    
    # Model parameters
    parser.add_argument('--hidden-size', type=int, default=64,
                      help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                      help='Starting epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                      help='Minimum epsilon value')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                      help='Epsilon decay rate')
    
    # Memory parameters
    parser.add_argument('--memory-size', type=int, default=10000,
                      help='Size of replay memory')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Training batch size')
    
    # Evaluation parameters
    parser.add_argument('--eval-episodes', type=int, default=10,
                      help='Number of evaluation episodes')
    parser.add_argument('--render-eval', action='store_true',
                      help='Render evaluation episodes')
    
    # Logging and saving
    parser.add_argument('--log-interval', type=int, default=100,
                      help='Episodes between logging')
    parser.add_argument('--save-interval', type=int, default=500,
                      help='Episodes between saving model')
    parser.add_argument('--exp-name', type=str, default=None,
                      help='Experiment name')
    parser.add_argument('--load-model', type=str, default=None,
                      help='Path to load model from')
    
    return parser.parse_args()

def setup_experiment_dir(args):
    """Set up experiment directory and save configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"dqn_experiment_{timestamp}"
    exp_dir = Path("experiments") / exp_name
    
    # Create directories
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    
    # Save configuration
    config_dict = vars(args)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    
    return exp_dir

def create_configs(args, exp_dir):
    """Create configuration objects from arguments."""
    dqn_config = DQNConfig(
        hidden_size=args.hidden_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        memory_size=args.memory_size,
        batch_size=args.batch_size
    )
    
    training_config = TrainingConfig(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        is_slippery=args.slippery,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        num_eval_episodes=args.eval_episodes,
        model_path=str(exp_dir / "models" / "dqn_model.pt"),
        render_mode="human" if args.render_eval else None
    )
    
    return dqn_config, training_config

def evaluate_policy(trainer, num_episodes, render=False):
    """Evaluate the current policy."""
    eval_rewards = []
    eval_steps = []
    
    for episode in range(num_episodes):
        state, _ = trainer.env.reset()
        episode_rewards = []
        episode_states = []
        episode_actions = []
        steps = 0
        
        while True:
            if render:
                trainer.env.render()
            
            # Store current state
            episode_states.append(state)
            
            # Get and store action
            action = trainer.agent.select_action(state)
            episode_actions.append(action)
            
            # Take step
            next_state, reward, terminated, truncated, _ = trainer.env.step(action)
            episode_rewards.append(reward)
            
            steps += 1
            state = next_state
            
            if terminated or truncated or steps >= trainer.config.max_steps:
                break
        
        eval_rewards.append(sum(episode_rewards))
        eval_steps.append(steps)
        
        # Plot episode visualization for first evaluation episode
        if episode == 0:
            plot_episode_visualization(
                episode_states,
                episode_actions,
                episode_rewards,
                save_path=str(Path(trainer.config.model_path).parent.parent / "plots" / "eval_episode.png")
            )
    
    return np.mean(eval_rewards), np.std(eval_rewards), np.mean(eval_steps)

def main():
    """Main function for training and evaluation."""
    args = parse_args()
    exp_dir = setup_experiment_dir(args)
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Create configurations
    dqn_config, training_config = create_configs(args, exp_dir)
    
    # Set up trainer
    trainer = FrozenLakeTrainer(dqn_config, training_config)
    
    # Load pre-trained model if specified
    if args.load_model:
        logger.info(f"Loading model from {args.load_model}")
        trainer.agent.load(args.load_model)
    
    # Train agent
    logger.info("Starting training...")
    try:
        rewards_history, steps_history = trainer.train()
        
        # Plot training metrics
        plot_training_metrics(
            rewards_history,
            steps_history,
            save_path=str(exp_dir / "plots" / "training_metrics.png")
        )
        
        # Plot Q-values
        plot_q_values(
            trainer.agent.policy_net.fc1.weight.detach().numpy(),
            save_path=str(exp_dir / "plots" / "q_values.png")
        )
        
        # Final evaluation
        logger.info("Evaluating final policy...")
        mean_reward, std_reward, mean_steps = evaluate_policy(
            trainer,
            args.eval_episodes,
            render=args.render_eval
        )
        
        # Save evaluation results
        eval_results = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "mean_steps": float(mean_steps)
        }
        with open(exp_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=4)
        
        logger.info(f"Final Evaluation - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}, "
                   f"Mean Steps: {mean_steps:.2f}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Save final model
        final_model_path = exp_dir / "models" / "final_model.pt"
        trainer.agent.save(str(final_model_path))
        logger.info(f"Saved final model to {final_model_path}")
        
        # Clean up
        trainer.close()

if __name__ == "__main__":
    main()