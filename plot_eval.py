import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import os
from evaluation import *
from baselines import *
METHOD_COLORS = {'Random': 'r', 'Sequential': 'g', 'DQN': 'b', 'Greedy': 'y', 'Genetic': 'purple', 'DQN_ALT': 'orange'}

def get_baseline_data(path):
    """
    Reads baseline data from a CSV file and calculates the mean waiting time.

    Args:
        path (str): Path to the CSV file containing baseline data.

    Returns:
        float: Mean waiting time for the baseline.
    """
    df = pd.read_csv(path)
    # Calculate the average waiting time for these last steps
    waiting_times = df['Average Waiting Time']
    mean_waiting_time = np.mean(waiting_times)
    return mean_waiting_time

def plot_dqn_data(base_path, baselines_values=None, show=True):
    """
    Plots DQN performance data and compares it with baselines.

    Args:
        base_path (str): Base path to DQN data files.
        baselines_values (dict, optional): Dictionary of baseline values for comparison.
        show (bool): Whether to display the plot or save it to a file.

    Returns:
        tuple: Lists of average waiting times for DQN and DQN with alternative reward.
    """
    i = 1
    average_waiting_times = []
    dqn_data_2nd_reward = []

    while True:
        file_name = f"dqn_conn0_ep{i}.csv"
        file_path = os.path.join(base_path, file_name)

        file_name_2nd_reward = f"second/dqn_conn0_ep{i}.csv"
        file_path_2nd_reward = os.path.join(base_path, file_name_2nd_reward)

        if not os.path.exists(file_path):
            break  # Exit the loop if the file doesn't exist
        
        df = pd.read_csv(file_path)
        df_2nd_reward = pd.read_csv(file_path_2nd_reward)

        waiting_times = df['system_mean_waiting_time']
        waiting_times_2nd_reward = df_2nd_reward['system_mean_waiting_time']

        average_waiting_times.append(np.mean(waiting_times))
        dqn_data_2nd_reward.append(np.mean(waiting_times_2nd_reward))
        i += 1
    


    fig, ax = plt.subplots(figsize=(12, 6))  # Increased width from default
    ax.plot(range(len(average_waiting_times)), average_waiting_times, color=METHOD_COLORS['DQN'], label='DQN')
    ax.plot(range(len(dqn_data_2nd_reward)), dqn_data_2nd_reward, color=METHOD_COLORS['DQN_ALT'], label='DQN with change in cumulative vehicle delay')
    
    if baselines_values:
        for baseline_name, baseline_value in baselines_values.items():
            ax.axhline(y=baseline_value, color=METHOD_COLORS[baseline_name], linestyle='--', label=baseline_name)
    
    ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Car Waiting Time (log scale)')
    ax.set_title('DQN - Average Car Waiting Time over Episodes')
    
    # Set y-axis to logarithmic scale
    ax.set_yscale('log')
    
    # Add some padding to the x and y limits
    x_padding = len(average_waiting_times) * 0.05
    y_values = average_waiting_times + dqn_data_2nd_reward + list(baselines_values.values())
    y_min, y_max = min(y_values), max(y_values)
    
    ax.set_xlim(0 - x_padding, len(average_waiting_times) - 1 + x_padding)
    ax.set_ylim(y_min * 0.9, y_max * 1.1)  # Adjust y-limits for log 

    plt.tight_layout()  # Adjust the layout to prevent cutoff
    if show:
        plt.show()
    else:
        # Plot the final state
        plt.savefig('outputs/2way-single-intersection/plots/dqn_plot_final.png')
        plt.close(fig)
    
    return average_waiting_times, dqn_data_2nd_reward


def plot_gen_data(path, baselines_values=None, show=True):
    """
    Plots Genetic Algorithm performance data and compares it with baselines.

    Args:
        path (str): Path to the Genetic Algorithm data file.
        baselines_values (dict, optional): Dictionary of baseline values for comparison.
        show (bool): Whether to display the plot or save it to a file.

    Returns:
        pandas.DataFrame: The loaded Genetic Algorithm data.
    """
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(12, 6))  # Increased width from default

    # Plot genetic algorithm data
    ax.plot(df['Generation'], df['Average Waiting Time(Fitness)'], color=METHOD_COLORS['Genetic'], label='Genetic')
    
    # Plot baselines
    if baselines_values:
        for baseline_name, baseline_value in baselines_values.items():
            ax.axhline(y=baseline_value, color=METHOD_COLORS[baseline_name], linestyle='--', label=baseline_name)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Car Waiting Time')
    ax.set_title('Genetic Algorithm - Average Car Waiting Time over Generations')
    
    # Add some padding to the x and y limits
    x_padding = (df['Generation'].max() - df['Generation'].min()) * 0.05
    y_values = df['Average Waiting Time(Fitness)'].tolist() + list(baselines_values.values())
    y_min, y_max = min(y_values), max(y_values)
    y_padding = (y_max - y_min) * 0.05
    
    x_min = df['Generation'].min() - x_padding
    x_max = df['Generation'].max() + x_padding
    y_min = min(y_min, min(baselines_values.values()) if baselines_values else float('inf')) - y_padding
    y_max = max(y_max, max(baselines_values.values()) if baselines_values else float('-inf')) + y_padding

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()  # Adjust the layout to prevent cutoff
    if show:
        plt.show()
    else:
        # Plot the final state
        plt.savefig('outputs/2way-single-intersection/plots/gen_plot_final.png')
        plt.close(fig)
    return df


def plot_combined_data(dqn_base_path, gen_path, baselines_values=None, show=True):
    """
    Plots combined DQN and Genetic Algorithm performance data with baselines.

    Args:
        dqn_base_path (str): Base path to DQN data files.
        gen_path (str): Path to the Genetic Algorithm data file.
        baselines_values (dict, optional): Dictionary of baseline values for comparison.
        show (bool): Whether to display the plot or save it to a file.
    """
    # Get DQN data
    dqn_data = []
    i = 1
    while True:
        file_name = f"dqn_conn0_ep{i}.csv"
        file_path = os.path.join(dqn_base_path, file_name)
        
        if not os.path.exists(file_path):
            break
        
        df = pd.read_csv(file_path)
        waiting_times = df['system_mean_waiting_time']
        dqn_data.append(np.mean(waiting_times))
        i += 1

    # Get Genetic Algorithm data
    gen_df = pd.read_csv(gen_path)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Set y-axis to logarithmic scale
    ax1.set_yscale('log')

    # Plot DQN data
    line_dqn, = ax1.plot(range(len(dqn_data)), dqn_data, color=METHOD_COLORS['DQN'], label='DQN')
    ax1.set_xlabel('DQN Episode')
    ax1.set_ylabel('Average Car Waiting Time (log scale)', color=METHOD_COLORS['DQN'])
    ax1.tick_params(axis='y', labelcolor=METHOD_COLORS['DQN'])

    # Create a secondary x-axis for Genetic Algorithm
    ax2 = ax1.twiny()
    line_gen, = ax2.plot(gen_df['Generation'], gen_df['Average Waiting Time(Fitness)'], color=METHOD_COLORS['Genetic'], label='Genetic')
    ax2.set_xlabel('Genetic Algorithm Generation', color=METHOD_COLORS['Genetic'])
    ax2.tick_params(axis='x', labelcolor=METHOD_COLORS['Genetic'])

    baseline_lines = []
    if baselines_values:
        for baseline_name, baseline_value in baselines_values.items():
            line = ax1.axhline(y=baseline_value, color=METHOD_COLORS[baseline_name], linestyle='--', label=baseline_name)
            baseline_lines.append(line)

    lines = [line_dqn, line_gen] + baseline_lines
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    # Set y-axis limits with some padding in log space
    all_y_values = dqn_data + gen_df['Average Waiting Time(Fitness)'].tolist() + list(baselines_values.values())
    y_min, y_max = min(all_y_values), max(all_y_values)
    ax1.set_ylim(y_min * 0.9, y_max * 1.1)

    plt.title('DQN vs Genetic Algorithm - Average Car Waiting Time (log scale)')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(f'outputs/2way-single-intersection/plots/combined_plot.png')

if __name__ == "__main__":
    baselines_values = {}
    for baseline in baselines:
        baseline_path = f'outputs/2way-single-intersection/baselines/{baseline}_evaluation.csv'
        baselines_values[baseline] = get_baseline_data(baseline_path)
    
    # Choose which plot to display by commenting out the others
    plot_dqn_data('outputs/2way-single-intersection/dqn/', baselines_values, show=True)
    plot_gen_data('outputs/2way-single-intersection/genetic/ga_evaluation.csv', baselines_values, show=True)

    # plots both dqn and genetic algorithm data
    plot_combined_data(
        'outputs/2way-single-intersection/dqn/',
        'outputs/2way-single-intersection/genetic/ga_evaluation.csv',
        baselines_values, show=True
    )
