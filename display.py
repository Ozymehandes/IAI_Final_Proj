import os
import sys
import numpy as np
from sumo_rl import SumoEnvironment
from sumo_rl.environment.traffic_signal import TrafficSignal
from stable_baselines3 import DQN, A2C, PPO
import csv
from evaluation import *
from gen import *
from baselines import *

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


def display_generation(env, base_path, generation_index):
    """
    Display a representative individual from a specific generation.

    Args:
        env: The SUMO environment.
        base_path (str): Path to the directory containing generation files.
        generation_index (int): Index of the generation to display.

    Returns:
        numpy.ndarray: The representative individual's action sequence.
    """
    generation_path = base_path + f'population_{generation_index}.npy'
    generation = np.load(generation_path)
    random_index = random.randint(0, len(generation) - 1)
    representative = generation[random_index]

    obs = env.reset()
    done = False
    for action in representative:
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            break
    return representative

def display_final(env, base_path):
    """
    Display the best individual from the final generation.

    Args:
        env: The SUMO environment.
        base_path (str): Path to the directory containing the final results.

    Returns:
        numpy.ndarray: The best individual's action sequence.
    """
    final_path = base_path + 'ga_results.npz'
    final_generation, final_fitnesses, best_individual, best_individual_index = load_genetic_results(final_path)
    best_individual = final_generation[-1]

    obs = env.reset()
    done = False
    for action in best_individual:
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            break
    return best_individual

def display_baseline(env, baseline_name):
    """
    Display a baseline model's performance.

    Args:
        env: The SUMO environment.
        baseline_name (str): Name of the baseline model to use.

    Returns:
        object: The baseline model.
    """
    model = baselines[baseline_name](env.action_space)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return model

def display_dqn(env, path=None, pretrained=False):
    """
    Display a DQN model's performance, either pretrained or newly initialized.

    Args:
        env: The SUMO environment.
        path (str, optional): Path to the pretrained model.
        pretrained (bool): Whether to use a pretrained model or create a new one.

    Returns:
        stable_baselines3.DQN: The DQN model.
    """
    if pretrained and path:
        model = DQN.load(path)
    else:
        model = DQN(env=env, policy="MlpPolicy", learning_rate=0.001, learning_starts=0, train_freq=1, target_update_interval=500, exploration_initial_eps=0.05, exploration_final_eps=0.01, verbose=1)

    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return model


if __name__ == "__main__":
    net_file = NET_FILE
    route_file = ROUTE_FILE
    out_csv_name = "outputs/2way-single-intersection/display/display"
    env = create_env_with_reward(net_file=net_file, route_file=route_file, out_csv_name=out_csv_name, reward_fn=system_mean_waiting_time_reward, use_gui=True)
    display_dqn(env, pretrained=True, path="outputs/2way-single-intersection/dqn/dqn_model.zip")

    k = 20 # Choose generation to run
    display_generation(env, "outputs/2way-single-intersection/genetic/", k)
    display_final(env, "outputs/2way-single-intersection/genetic/")


