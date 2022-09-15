from typing import Tuple, List, Union

from pymoo.indicators.hv import HV
import numpy as np


def hypervolume(ref_point: np.ndarray, points: List[np.ndarray]) -> float:
    """Computes the hypervolume metric for a set of points (value vectors) and a reference point.

    Args:
        ref_point (np.ndarray): Reference point
        points (List[np.ndarray]): List of value vectors

    Returns:
        float: Hypervolume metric
    """    
    return HV(ref_point=ref_point * - 1)(np.array(points) * - 1)


def eval_mo(agent, env, w: np.ndarray, render: bool = False) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates one episode of the agent in the environment.

    Args:
        agent: Agent
        env: MO-Gym environment with LinearReward wrapper
        w (np.ndarray): Weight vector
        render (bool, optional): Whether to render the environment. Defaults to False.

    Returns:
        (float, float, np.ndarray, np.ndarray): Scalarized total reward, scalarized return, vectorized total reward, vectorized return
    """    
    obs = env.reset()
    done = False
    total_vec_reward, vec_return = np.zeros_like(w), np.zeros_like(w)
    gamma = 1.0
    while not done:
        if render:
            env.render(mode='human')
        obs, r, done, info = env.step(agent.eval(obs, w))
        total_vec_reward += r
        vec_return += gamma * r
        gamma *= agent.gamma
    return (
        np.dot(w, total_vec_reward),
        np.dot(w, vec_return),
        total_vec_reward,
        vec_return,
    )


def policy_evaluation_mo(agent, env, w: np.ndarray, rep: int = 5, return_scalarized_value: bool = False) -> Union[np.ndarray, float]:
    """Evaluates the value of a policy by runnning the policy for multiple episodes.

    Args:
        agent: Agent
        env: MO-Gym environment with LinearReward wrapper
        w (np.ndarray): Weight vector
        rep (int, optional): Number of episodes for averaging. Defaults to 5.
        return_scalarized_value (bool, optional): Whether to return scalarized value. Defaults to False.

    Returns:
        np.ndarray: Value of the policy
    """    
    if return_scalarized_value:
        returns = [eval_mo(agent, env, w)[1] for _ in range(rep)]
    else:
        returns = [eval_mo(agent, env, w)[3] for _ in range(rep)]
    return np.mean(returns, axis=0)