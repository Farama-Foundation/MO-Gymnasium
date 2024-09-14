import os
import sys


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import gymnasium as gym
import numpy as np
from sumo_rl import SumoEnvironment


def mo_reward_function(self):
    """MO reward function."""
    edges = {p: self.lanes[p * 4 : (p + 1) * 4] for p in range(self.num_green_phases)}

    def get_veh_list(p):
        veh_list = []
        for lane in edges[p]:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    wait_time_per_road = []
    for p in range(self.num_green_phases):  # self.num_green_phases 4 (direction)
        veh_list = get_veh_list(p)
        wait_time = 0.0
        for veh in veh_list:
            veh_lane = self.sumo.vehicle.getLaneID(veh)
            acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
            if veh not in self.env.vehicles:
                self.env.vehicles[veh] = {veh_lane: acc}
            else:
                self.env.vehicles[veh][veh_lane] = acc - sum(
                    [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                )
            wait_time += self.env.vehicles[veh][veh_lane]
        wait_time_per_road.append(wait_time)

    ts_wait = np.array(wait_time_per_road, dtype=np.float32) / 1000.0  # scale
    self.last_measure = ts_wait
    return -ts_wait


class MOIntersectionEnv(SumoEnvironment):
    """
    ## Description
    Multi-objective traffic signal control using SUMO-RL (https://github.com/LucasAlegre/sumo-rl).
    This environment correspond to a big intersection, as defined in:
    The Max-Min Formulation of Multi-Objective Reinforcement Learning: From Theory to a Model-Free Algorithm
    Giseung Park and Woohyeon Byeon and Seongmin Kim and Elad Havakuk and Amir Leshem and Youngchul
    Forty-first International Conference on Machine Learning (ICML), 2024.

    ## Observation Space
    See https://lucasalegre.github.io/sumo-rl/mdp/observation/.

    ## Action Space
    The action space is discrete and corresponds to the traffic signal phases.
    See https://lucasalegre.github.io/sumo-rl/mdp/action/.

    ## Reward Space
    The reward is 4-dimensional, corresponding to the negative of the average waiting time per road.

    ## Episode Termination
    The episode terminates after 9000 seconds. There is no terminal state.

    ## Credits
    Environment from the SUMO-RL library (https://github.com/LucasAlegre/sumo-rl).
    Intersection defined as in https://github.com/Giseung-Park/Maxmin-MORL.
    """

    def __init__(self, *args, **kwargs):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        super().__init__(
            single_agent=True,
            net_file=dir_path + "/assets/big_intersection.net.xml",
            route_file=dir_path + "/assets/routes.rou.xml",
            yellow_time=4,
            delta_time=30,
            num_seconds=9000,
            reward_fn=mo_reward_function,
            *args,
            **kwargs,
        )
        self.reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        self.reward_dim = 4
