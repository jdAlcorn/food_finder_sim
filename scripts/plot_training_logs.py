#!/usr/bin/env python3
"""
Plot training logs from RL GRU training CSV files

This script creates various plots to visualize training progress:
- Reward trends over time
- Success rate progression
- Training loss curves
- Food collection statistics
- Test case distribution (if using scheduler)
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_training_data(csv_path: str) -> pd.DataFrame:
    """Load training data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} episodes from {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def plot_reward_trends(df: pd.DataFrame, save_dir: str = None):
    """Plot reward trends over training"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Reward Trends During Training', fontsize=16)
    
    # Episode reward over time
    axes[0, 0].plot(df['episode'], df['reward'], alpha=0.6, linewidth=0.8, label='Episode Reward')
    axes[0, 0].plot(df['episode'], df['avg_reward_100'], linewidth=2, label='100-Episode Average')
    axes[0, 0].plot(df['episode'], df['best_reward'], linewidth=2, label='Best Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Reward Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Success rate over time
    axes[0, 1].plot(df['episode'], df['success_rate_100'] * 100, linewidth=2, color='green')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate (%)')
    axes[0, 1].set_title('Success Rate (100-Episode Window)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 100)
    
    # Episode length over time
    axes[1, 0].plot(df['episode'], df['episode_length'], alpha=0.6, linewidth=0.8)
    # Add rolling average
    window = min(50, len(df) // 10)
    if window > 1:
        rolling_length = df['episode_length'].rolling(window=window, center=True).mean()
        axes[1, 0].plot(df['episode'], rolling_length, linewidth=2, color='red', label=f'{window}-Episode Average')
        axes[1, 0].legend()
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Episode Length')
    axes[1, 0].set_title('Episode Length Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reward components breakdown
    axes[1, 1].plot(df['episode'], df['total_terminal_reward'], label='Terminal', linewidth=1.5)
    axes[1, 1].plot(df['episode'], df['total_reacquire_bonus'], label='Reacquire', linewidth=1.5)
    axes[1, 1].plot(df['episode'], df['total_progress_reward'], label='Progress', linewidth=1.5)
    axes[1, 1].plot(df['episode'], df['total_step_penalty'], label='Step Penalty', linewidth=1.5)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward Component')
    axes[1, 1].set_title('Reward Components Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'reward_trends.png'), dpi=300, bbox_inches='tight')
        print(f"Saved reward trends plot to {save_dir}/reward_trends.png")
    
    plt.show()


def plot_training_metrics(df: pd.DataFrame, save_dir: str = None):
    """Plot training loss and gradient metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Policy and value loss
    axes[0, 0].plot(df['episode'], df['policy_loss'], label='Policy Loss', linewidth=1.5)
    axes[0, 0].plot(df['episode'], df['value_loss'], label='Value Loss', linewidth=1.5)
    axes[0, 0].plot(df['episode'], df['total_loss'], label='Total Loss', linewidth=1.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gradient norms
    axes[0, 1].plot(df['episode'], df['grad_norm_pre_clip'], label='Pre-Clip', linewidth=1.5)
    axes[0, 1].plot(df['episode'], df['grad_norm_post_clip'], label='Post-Clip', linewidth=1.5)
    axes[0, 1].plot(df['episode'], df['gru_grad_norm'], label='GRU Only', linewidth=1.5)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].set_title('Gradient Norms')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Entropy over time
    axes[1, 0].plot(df['episode'], df['entropy'], linewidth=2, color='purple')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].set_title('Policy Entropy Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Food visibility and distance metrics
    # Only plot if we have valid data (not all -1)
    if (df['min_food_distance_seen'] > -1).any():
        valid_distances = df[df['min_food_distance_seen'] > -1]
        axes[1, 1].scatter(valid_distances['episode'], valid_distances['min_food_distance_seen'], 
                          alpha=0.6, s=10, label='Min Distance Seen')
        
    if (df['final_food_distance'] > -1).any():
        valid_final = df[df['final_food_distance'] > -1]
        axes[1, 1].scatter(valid_final['episode'], valid_final['final_food_distance'], 
                          alpha=0.6, s=10, label='Final Distance')
    
    axes[1, 1].plot(df['episode'], df['food_visible_fraction'] * 100, 
                   linewidth=1.5, color='orange', label='Food Visible %')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Distance / Percentage')
    axes[1, 1].set_title('Food Visibility and Distance Metrics')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        print(f"Saved training metrics plot to {save_dir}/training_metrics.png")
    
    plt.show()


def plot_test_case_performance_over_time(df: pd.DataFrame, save_dir: str = None):
    """Plot success rate over time for each test case"""
    unique_cases = df['test_case_id'].nunique()
    
    if unique_cases == 1:
        print(f"Only one test case used: {df['test_case_id'].iloc[0]} - skipping per-case performance plot")
        return
    
    # Calculate rolling success rate for each test case
    window_size = max(10, len(df) // 50)  # Adaptive window size
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle(f'Test Case Performance Over Time ({unique_cases} cases)', fontsize=16)
    
    # Plot 1: Success over episode number for each test case
    colors = plt.cm.tab10(np.linspace(0, 1, unique_cases))
    
    for i, case_id in enumerate(df['test_case_id'].unique()):
        case_data = df[df['test_case_id'] == case_id].copy()
        
        if len(case_data) < 5:  # Skip cases with too few episodes
            continue
            
        # Plot raw success points
        success_episodes = case_data[case_data['success'] == 1]['episode']
        failure_episodes = case_data[case_data['success'] == 0]['episode']
        
        axes[0].scatter(success_episodes, [i] * len(success_episodes), 
                       color=colors[i], marker='o', s=20, alpha=0.7, label=f'{case_id} (Success)')
        axes[0].scatter(failure_episodes, [i] * len(failure_episodes), 
                       color=colors[i], marker='x', s=20, alpha=0.4)
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Test Case')
    axes[0].set_title('Success/Failure by Episode (○ = Success, × = Failure)')
    axes[0].set_yticks(range(unique_cases))
    axes[0].set_yticklabels([case[:20] + '...' if len(case) > 20 else case 
                            for case in df['test_case_id'].unique()])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Rolling success rate over time for each test case
    for i, case_id in enumerate(df['test_case_id'].unique()):
        case_data = df[df['test_case_id'] == case_id].copy()
        
        if len(case_data) < window_size:  # Skip cases with insufficient data
            continue
        
        # Calculate rolling success rate
        case_data = case_data.sort_values('episode')
        rolling_success = case_data['success'].rolling(window=min(window_size, len(case_data)), 
                                                      center=True, min_periods=1).mean()
        
        axes[1].plot(case_data['episode'], rolling_success * 100, 
                    color=colors[i], linewidth=2, label=f'{case_id} (n={len(case_data)})')
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].set_title(f'Rolling Success Rate Over Time (window={window_size})')
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'test_case_performance_over_time.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Saved test case performance over time plot to {save_dir}/test_case_performance_over_time.png")
    
    plt.show()


def plot_test_case_distribution(df: pd.DataFrame, save_dir: str = None):
    """Plot test case distribution if using scheduler"""
    unique_cases = df['test_case_id'].nunique()
    
    if unique_cases == 1:
        print(f"Only one test case used: {df['test_case_id'].iloc[0]}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Test Case Distribution ({unique_cases} different cases)', fontsize=16)
    
    # Test case frequency
    case_counts = df['test_case_id'].value_counts()
    axes[0].pie(case_counts.values, labels=case_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Test Case Usage Distribution')
    
    # Success rate by test case
    success_by_case = df.groupby('test_case_id')['success'].agg(['mean', 'count']).reset_index()
    success_by_case = success_by_case[success_by_case['count'] >= 5]  # Only cases with 5+ episodes
    
    if len(success_by_case) > 0:
        bars = axes[1].bar(range(len(success_by_case)), success_by_case['mean'] * 100)
        axes[1].set_xlabel('Test Case')
        axes[1].set_ylabel('Success Rate (%)')
        axes[1].set_title('Success Rate by Test Case')
        axes[1].set_xticks(range(len(success_by_case)))
        axes[1].set_xticklabels([case[:15] + '...' if len(case) > 15 else case 
                                for case in success_by_case['test_case_id']], rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, success_by_case['count'])):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'n={count}', ha='center', va='bottom', fontsize=8)
    else:
        axes[1].text(0.5, 0.5, 'Insufficient data\n(need 5+ episodes per case)', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Success Rate by Test Case')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'test_case_distribution.png'), dpi=300, bbox_inches='tight')
        print(f"Saved test case distribution plot to {save_dir}/test_case_distribution.png")
    
    plt.show()


def print_training_summary(df: pd.DataFrame):
    """Print a summary of the training run"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    total_episodes = len(df)
    final_avg_reward = df['avg_reward_100'].iloc[-1]
    best_reward = df['best_reward'].max()
    final_success_rate = df['success_rate_100'].iloc[-1] * 100
    total_time = df['training_time_elapsed'].iloc[-1] / 3600  # Convert to hours
    
    print(f"Total Episodes: {total_episodes}")
    print(f"Final Average Reward (100-ep): {final_avg_reward:.2f}")
    print(f"Best Single Episode Reward: {best_reward:.2f}")
    print(f"Final Success Rate: {final_success_rate:.1f}%")
    print(f"Total Training Time: {total_time:.2f} hours")
    
    # Test case info
    unique_cases = df['test_case_id'].nunique()
    if unique_cases > 1:
        print(f"Test Cases Used: {unique_cases}")
        most_common_case = df['test_case_id'].mode().iloc[0]
        case_percentage = (df['test_case_id'] == most_common_case).mean() * 100
        print(f"Most Common Case: {most_common_case} ({case_percentage:.1f}%)")
    else:
        print(f"Single Test Case: {df['test_case_id'].iloc[0]}")
    
    # Performance trends
    if total_episodes >= 100:
        early_success = df['success_rate_100'].iloc[99]  # Episode 100's success rate
        late_success = df['success_rate_100'].iloc[-1]   # Final success rate
        improvement = (late_success - early_success) * 100
        print(f"Success Rate Improvement (ep 100 → final): {improvement:+.1f}%")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Plot RL training logs from CSV files')
    parser.add_argument('csv_path', type=str, help='Path to training CSV log file')
    parser.add_argument('--save-plots', action='store_true', 
                       help='Save plots to PNG files in the same directory as CSV')
    parser.add_argument('--no-show', action='store_true',
                       help='Don\'t display plots interactively (useful for batch processing)')
    
    args = parser.parse_args()
    
    # Load data
    df = load_training_data(args.csv_path)
    if df is None:
        return
    
    # Determine save directory
    save_dir = None
    if args.save_plots:
        save_dir = os.path.dirname(args.csv_path)
        print(f"Will save plots to: {save_dir}")
    
    # Set matplotlib backend for non-interactive mode
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    
    # Print summary
    print_training_summary(df)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_reward_trends(df, save_dir)
    plot_training_metrics(df, save_dir)
    plot_test_case_performance_over_time(df, save_dir)
    plot_test_case_distribution(df, save_dir)
    
    print("\nPlotting complete!")


if __name__ == "__main__":
    main()