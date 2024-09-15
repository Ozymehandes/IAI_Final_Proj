from sumo_rl.environment.env import SumoEnvironment
import numpy as np

N_SECONDS = 3600
SEED = 42

# Change these to run different simulations
NET_FILE = '2way-single-intersection-edited/single-intersection.net.xml'
ROUTE_FILE = '2way-single-intersection-edited/single-intersection-vhvh.rou.xml'


def system_mean_waiting_time_reward(traffic_signal):
    """
    Calculates the reward based on the negative of the system's mean waiting time.

    Args:
        traffic_signal: The traffic signal object from the SUMO environment.

    Returns:
        float: The negative of the system's mean waiting time.
    """
    return -traffic_signal.env._get_system_info()['system_mean_waiting_time']

def create_env_with_reward(net_file, route_file, out_csv_name=None, use_gui=False, num_seconds=N_SECONDS, reward_fn=system_mean_waiting_time_reward, fixed_ts=False):
    """
    Creates a SUMO environment with specified parameters and reward function.

    Args:
        net_file (str): Path to the SUMO network file.
        route_file (str): Path to the SUMO route file.
        out_csv_name (str, optional): Name of the output CSV file. Defaults to None.
        use_gui (bool): Whether to use the SUMO GUI. Defaults to False.
        num_seconds (int): Number of seconds to run the simulation. Defaults to N_SECONDS.
        reward_fn (function): The reward function to use. Defaults to system_mean_waiting_time_reward.
        fixed_ts (bool): Whether to use fixed time steps. Defaults to False.

    Returns:
        SumoEnvironment: A configured SUMO environment instance.
    """
    return SumoEnvironment(net_file=net_file, route_file=route_file, out_csv_name=out_csv_name, use_gui=use_gui, single_agent=True,
                            num_seconds=num_seconds, reward_fn=reward_fn, fixed_ts=fixed_ts)
