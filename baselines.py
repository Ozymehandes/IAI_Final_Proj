import os
import sys
import numpy as np
from sumo_rl import SumoEnvironment
from sumo_rl.environment.traffic_signal import TrafficSignal
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import csv
from evaluation import *

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

LANES_PER_GREEN = {0 : [0, 4], 1 : [1, 5], 2 : [2, 6], 3 : [3, 7]}

class GreedyModel:
    """
    A greedy model that selects the action with the longest queue.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=True):
        """
        Predict the action based on the longest queue.
        
        Args:
            observation (np.array): The current state observation.
            deterministic (bool): Not used in this model.
        
        Returns:
            tuple: The predicted action, None(for environment use)
        """
        queues = observation[13:]
        green_0_queue = queues[LANES_PER_GREEN[0][0]] + queues[LANES_PER_GREEN[0][1]]
        green_1_queue = queues[LANES_PER_GREEN[1][0]] + queues[LANES_PER_GREEN[1][1]]
        green_2_queue = queues[LANES_PER_GREEN[2][0]] + queues[LANES_PER_GREEN[2][1]]
        green_3_queue = queues[LANES_PER_GREEN[3][0]] + queues[LANES_PER_GREEN[3][1]]
        queue_list = [green_0_queue, green_1_queue, green_2_queue, green_3_queue]
        longest_queue = np.argmax(queue_list)
        return longest_queue, None

# Sequential baseline model
class SequentialModel:
    """
    A sequential model that cycles through actions in a set order.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=True):
        return self.action_space.sample(), None

# Random baseline model
class RandomModel:
    """
    A random model that selects actions randomly.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=True):
        return np.random.randint(0, self.action_space.n), None
    
baselines = {
    "Sequential": SequentialModel,
    "Random": RandomModel,
    "Greedy": GreedyModel,
}

# Evaluate function for baseline models
def evaluate_baseline(model, env, n_eval_episodes=10, model_name='', path=''):
    """
    Evaluate a baseline model over multiple episodes.

    Args:
        model: The baseline model to evaluate.
        env: The environment to run the evaluation in.
        n_eval_episodes (int): Number of episodes to evaluate.
        model_name (str): Name of the model for logging purposes.
        path (str): Path to save the evaluation results.

    Returns:
        tuple: Mean and standard deviation of total wait times across episodes.
    """
    episode_avg_wait_times = []

    # Open a CSV file to write the results
    with open(f'{path}/{model_name}_evaluation.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Episode', 'Average Waiting Time'])  # Write header
        
    for eps in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        total_mean_wait_time = 0
        step = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_mean_wait_time += info['system_mean_waiting_time']
            step += 1
        

        ep_avg_wait_time = total_mean_wait_time / step if step > 0 else 0
        episode_avg_wait_times.append(ep_avg_wait_time)
        with open(f'{path}/{model_name}_evaluation.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([eps + 1, ep_avg_wait_time])
    
    mean_total_wait_time = np.mean(episode_avg_wait_times)
    std_total_wait_time = np.std(episode_avg_wait_times)    
    return mean_total_wait_time, std_total_wait_time

def run_baseline(net_file, route_file, path='', use_gui=False, num_seconds=N_SECONDS, episodes=1, model_name=''):
    """
    Run a single episode of a baseline model.

    Args:
        net_file (str): Path to the SUMO network file.
        route_file (str): Path to the SUMO route file.
        path (str): Path to save the results.
        use_gui (bool): Whether to use SUMO GUI.
        num_seconds (int): Number of seconds to run the simulation.
        episodes (int): Number of episodes to run (always 1 in this function).
        model_name (str): Name of the baseline model to run.

    Returns:
        float: Average waiting time for the episode.
    """
    env = create_env_with_reward(net_file=net_file, route_file=route_file, out_csv_name=path + model_name, reward_fn=system_mean_waiting_time_reward)
    model = baselines[model_name](env.action_space)

    obs, info = env.reset()
    done = False
    total_mean_wait_time = 0
    step = 0
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_mean_wait_time += info['system_mean_waiting_time']
        step += 1
    
    ep_avg_wait_time = total_mean_wait_time / step if step > 0 else 0

    print(f"{model_name} Baseline:")
    print(f"  Mean total accumulated waiting time: {ep_avg_wait_time:.2f}")
    print()
    return ep_avg_wait_time


def evaluate_baselines(net_file, route_file, path='', use_gui=False, num_seconds=N_SECONDS, episodes=10):
    """
    Evaluate all baseline models.

    Args:
        net_file (str): Path to the SUMO network file.
        route_file (str): Path to the SUMO route file.
        path (str): Path to save the results.
        use_gui (bool): Whether to use SUMO GUI.
        num_seconds (int): Number of seconds to run each simulation.
        episodes (int): Number of episodes to evaluate each model.
    """
    np.random.seed(SEED)
    for model_name, model_class in baselines.items():
        if model_name == 'Sequential':
            env = create_env_with_reward(net_file=net_file, route_file=route_file, out_csv_name=path + model_name, reward_fn=system_mean_waiting_time_reward, fixed_ts=True, use_gui=use_gui)
        else:
            env = create_env_with_reward(net_file=net_file, route_file=route_file, out_csv_name=path + model_name, reward_fn=system_mean_waiting_time_reward, use_gui=use_gui)
        model = model_class(env.action_space)
        mean_total_wait_time, std_total_wait_time = evaluate_baseline(model, env, model_name=model_name, path=path, n_eval_episodes=episodes)

        print(f"{model_name} Baseline:")
        print(f"  Mean total accumulated waiting time: {mean_total_wait_time:.2f} +/- {std_total_wait_time:.2f}")
        print()

if __name__ == "__main__":
    # Running this will overwrite current results
    path = 'outputs/2way-single-intersection/baselines/'
    net_file = NET_FILE
    route_file = ROUTE_FILE
    evaluate_baselines(net_file, route_file, path, use_gui=False, num_seconds=N_SECONDS, episodes=10)
