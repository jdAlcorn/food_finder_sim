#!/usr/bin/env python3
"""
Plot training logs from RL MLP training CSV files

This script creates various plots to visualize training progress for the MLP trainer
with the focused reward system:
- Reward trends over time
- Success rate progression
- Training loss curves
- Food collection statistics
- Focused reward component analysis
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Reward Trends During Training (MLP Policy)', fontsize=16)
    
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
    
    # Focused reward components breakdown
    axes[1, 1].plot(df['episode'], df['total_terminal_reward'], label='Terminal', linewidth=1.5)
    axes[1, 1].plot(df['episode'], df['total_reacquire_reward'], label='Reacquire', linewidth=1.5)
    axes[1, 1].plot(df['episode'], df['total_center_progress'], label='Center Progress', linewidth=1.5)
    axes[1, 1].plot(df['episode'], df['total_distance_progress'], label='Distance Progress', linewidth=1.5)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward Component')
    axes[1, 1].set_title('Positive Reward Components Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'reward_trends.png'), dpi=300, bbox_inches='tight')
        print(f"Saved reward trends plot to {save_dir}/reward_trends.png")
    
    plt.show()


def plot_focused_reward_analysis(df: pd.DataFrame, save_dir: str = None):
    """Plot detailed analysis of the focused reward system components"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Focused Reward System Analysis', fontsize=16)
    
    # Terminal rewards (food collection events)
    axes[0, 0].plot(df['episode'], df['total_terminal_reward'], linewidth=2, color='gold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Terminal Reward')
    axes[0, 0].set_title('Terminal Rewards (Food Collection)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reacquire rewards (seeing food after losing it)
    axes[0, 1].plot(df['episode'], df['total_reacquire_reward'], linewidth=2, color='blue')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reacquire Reward')
    axes[0, 1].set_title('Reacquire Rewards (Finding Food Again)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Center progress rewards (food moving toward center of vision)
    axes[0, 2].plot(df['episode'], df['total_center_progress'], linewidth=2, color='purple')
    axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Center Progress Reward')
    axes[0, 2].set_title('Center Progress (Centering Food in Vision)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Distance progress rewards (getting closer to food)
    axes[1, 0].plot(df['episode'], df['total_distance_progress'], linewidth=2, color='green')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Distance Progress Reward')
    axes[1, 0].set_title('Distance Progress (Getting Closer to Food)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Penalties (negative rewards)
    axes[1, 1].plot(df['episode'], df['total_lost_sight_penalty'], label='Lost Sight', linewidth=2, color='red')
    axes[1, 1].plot(df['episode'], df['total_no_food_penalty'], label='No Food', linewidth=2, color='orange')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Penalty (Negative Reward)')
    axes[1, 1].set_title('Penalties Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Reward component balance (stacked area chart)
    # Prepare data for stacked area chart
    positive_components = ['total_terminal_reward', 'total_reacquire_reward', 
                          'total_center_progress', 'total_distance_progress']
    negative_components = ['total_lost_sight_penalty', 'total_no_food_penalty']
    
    # Only include positive values for positive components
    pos_data = df[positive_components].clip(lower=0)
    # Only include negative values (as positive for display) for negative components  
    neg_data = -df[negative_components].clip(upper=0)
    
    axes[1, 2].stackplot(df['episode'], pos_data.T, labels=positive_components, alpha=0.7)
    axes[1, 2].stackplot(df['episode'], -neg_data.T, labels=negative_components, alpha=0.7)
    axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.8)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Cumulative Reward Components')
    axes[1, 2].set_title('Reward Component Balance')
    axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'focused_reward_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"Saved focused reward analysis plot to {save_dir}/focused_reward_analysis.png")
    
    plt.show()


def plot_training_metrics(df: pd.DataFrame, save_dir: str = None):
    """Plot training loss and gradient metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Metrics (MLP Policy)', fontsize=16)
    
    # Policy and value loss
    axes[0, 0].plot(df['episode'], df['policy_loss'], label='Policy Loss', linewidth=1.5)
    axes[0, 0].plot(df['episode'], df['value_loss'], label='Value Loss', linewidth=1.5)
    axes[0, 0].plot(df['episode'], df['total_loss'], label='Total Loss', linewidth=1.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gradient norms (MLP doesn't have separate GRU grad norm)
    axes[0, 1].plot(df['episode'], df['grad_norm_pre_clip'], label='Pre-Clip', linewidth=1.5)
    axes[0, 1].plot(df['episode'], df['grad_norm_post_clip'], label='Post-Clip', linewidth=1.5)
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
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
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
    """Plot test case success rates and distribution"""
    unique_cases = df['test_case_id'].nunique()
    
    if unique_cases == 1:
        print(f"Only one test case used: {df['test_case_id'].iloc[0]}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Test Case Analysis ({unique_cases} different cases)', fontsize=16)
    
    # Calculate success rates and counts by test case
    case_stats = df.groupby('test_case_id').agg({
        'success': ['mean', 'sum', 'count']
    }).reset_index()
    case_stats.columns = ['test_case_id', 'success_rate', 'successes', 'total_episodes']
    
    # Filter out cases with very few episodes for the pie chart
    case_stats_filtered = case_stats[case_stats['total_episodes'] >= 3]
    
    if len(case_stats_filtered) == 0:
        axes[0].text(0.5, 0.5, 'Insufficient data\n(need 3+ episodes per case)', 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Success Rate Distribution')
    else:
        # Success rate pie chart
        success_rates = case_stats_filtered['success_rate'] * 100
        labels = [f"{case[:15]}{'...' if len(case) > 15 else ''}\n{rate:.1f}%" 
                 for case, rate in zip(case_stats_filtered['test_case_id'], success_rates)]
        
        # Color by success rate (green = high success, red = low success)
        colors = plt.cm.RdYlGn(success_rates / 100)  # Normalize to 0-1 for colormap
        
        wedges, texts, autotexts = axes[0].pie(case_stats_filtered['total_episodes'], 
                                              labels=labels, autopct='', startangle=90,
                                              colors=colors)
        axes[0].set_title('Success Rate by Test Case\n(Size = Episode Count)')
        
        # Add a color bar to show the success rate scale
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=100))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[0], shrink=0.8)
        cbar.set_label('Success Rate (%)')
    
    # Success rate bar chart (keep this as it's useful)
    if len(case_stats) > 0:
        # Sort by success rate for better visualization
        case_stats_sorted = case_stats.sort_values('success_rate', ascending=True)
        
        bars = axes[1].barh(range(len(case_stats_sorted)), case_stats_sorted['success_rate'] * 100)
        axes[1].set_xlabel('Success Rate (%)')
        axes[1].set_ylabel('Test Case')
        axes[1].set_title('Success Rate by Test Case')
        axes[1].set_yticks(range(len(case_stats_sorted)))
        axes[1].set_yticklabels([case[:20] + '...' if len(case) > 20 else case 
                                for case in case_stats_sorted['test_case_id']])
        axes[1].grid(True, alpha=0.3, axis='x')
        axes[1].set_xlim(0, 100)
        
        # Color bars by success rate
        for bar, rate in zip(bars, case_stats_sorted['success_rate'] * 100):
            bar.set_color(plt.cm.RdYlGn(rate / 100))
        
        # Add episode count labels on bars
        for i, (bar, count) in enumerate(zip(bars, case_stats_sorted['total_episodes'])):
            axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                        f'n={count}', ha='left', va='center', fontsize=8)
    else:
        axes[1].text(0.5, 0.5, 'No data available', 
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
    print("MLP TRAINING SUMMARY")
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
    
    # Focused reward system analysis
    print("\nFocused Reward System Analysis:")
    final_episode = df.iloc[-1]
    print(f"  Terminal Rewards: {final_episode['total_terminal_reward']:.1f}")
    print(f"  Reacquire Rewards: {final_episode['total_reacquire_reward']:.1f}")
    print(f"  Center Progress: {final_episode['total_center_progress']:.1f}")
    print(f"  Distance Progress: {final_episode['total_distance_progress']:.1f}")
    print(f"  Lost Sight Penalties: {final_episode['total_lost_sight_penalty']:.1f}")
    print(f"  No Food Penalties: {final_episode['total_no_food_penalty']:.1f}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Plot MLP RL training logs from CSV files')
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
    plot_focused_reward_analysis(df, save_dir)
    plot_training_metrics(df, save_dir)
    plot_test_case_performance_over_time(df, save_dir)
    plot_test_case_distribution(df, save_dir)
    
    print("\nPlotting complete!")


if __name__ == "__main__":
    main()