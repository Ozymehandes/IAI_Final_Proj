import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import csv
from stable_baselines3.dqn.dqn import DQN

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment
from evaluation import *


def train_dqn(net_file, route_file, out_csv_name=None, path='', use_gui=False, num_seconds=N_SECONDS):
    """
    Train a DQN model for traffic signal control.

    Args:
        net_file (str): Path to the SUMO network file.
        route_file (str): Path to the SUMO route file.
        out_csv_name (str, optional): Name of the output CSV file.
        path (str, optional): Path to save the trained model.
        use_gui (bool, optional): Whether to use SUMO GUI during training.
        num_seconds (int, optional): Number of seconds to run the simulation.

    Returns:
        None
    """
    env = create_env_with_reward(net_file=net_file, route_file=route_file, out_csv_name=out_csv_name, reward_fn=system_mean_waiting_time_reward)
    model = DQN(env=env, policy="MlpPolicy", learning_rate=0.001, learning_starts=0, train_freq=1, target_update_interval=500, exploration_initial_eps=0.05, exploration_final_eps=0.01, verbose=1)
    model.learn(total_timesteps=100000)
    model.save(f"{path}dqn_model")

def load_dqn(net_file, route_file, out_csv_name=None, path='', use_gui=False, num_seconds=N_SECONDS):
    """
    Load a pre-trained DQN model for traffic signal control.

    Args:
        net_file (str): Path to the SUMO network file.
        route_file (str): Path to the SUMO route file.
        out_csv_name (str, optional): Name of the output CSV file.
        path (str, optional): Path to load the trained model from.
        use_gui (bool, optional): Whether to use SUMO GUI during evaluation.
        num_seconds (int, optional): Number of seconds to run the simulation.

    Returns:
        tuple: A tuple containing the environment and the loaded DQN model.
    """
    env = create_env_with_reward(net_file=net_file, route_file=route_file, out_csv_name=out_csv_name, reward_fn=system_mean_waiting_time_reward)
    model = DQN.load(f"{path}dqn_model", env=env)
    return env, model

if __name__ == "__main__":
    # Running this will overwrite current results, change paths to avoid this
    train_dqn(NET_FILE, ROUTE_FILE, out_csv_name='outputs/2way-single-intersection/dqn/dqn', use_gui=False, path='outputs/2way-single-intersection/dqn/')
    
    # Load the model for further examination
    env, model = load_dqn(NET_FILE, ROUTE_FILE, out_csv_name='outputs/2way-single-intersection/dqn/dqn', use_gui=False, path='outputs/2way-single-intersection/dqn/')
