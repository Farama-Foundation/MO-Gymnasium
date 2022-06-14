from gym import Env
import gym.spaces as spaces
import numpy as np

class FoodWaterBalance(Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    # actions
    LEFT = 0
    RIGHT = 1

    def __init__(self, num_states=3, 
                #gamma=0.9,
                gamma_past=0.99,
                max_steps=1e5, 
                use_more=False):
        self.num_observations = num_states
        self.num_actions = 2
        self.num_objectives = 2

        #self.gamma = gamma
        self.gamma_past = gamma_past
        self.max_steps = max_steps
        self.use_more = use_more

        # initial state
        self.state = 0

        if self.use_more:
            # dynamic weights used in M.O.R.E
            self.more_weights = 0.5*np.ones(self.num_objectives)

        # past cumulative reward
        #self.cum_reward = np.zeros(self.num_objectives)

        # define spaces
        self.reward_space = spaces.Box(np.array([-0.018, -0.09]), np.array([0.1, 0.02]))
        #self.min_cum_reward = self.reward_space.low/(1 - self.gamma)
        #self.max_cum_reward = self.reward_space.high/(1 - self.gamma)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3) 
                                #spaces.Dict({"discrete_state": spaces.Discrete(3),
                                #             "cum_rewards": spaces.Box(self.min_cum_reward, self.max_cum_reward)})

        self.step_counter = 0

    def reset(self):
        self.state = 0
        self.cum_reward = np.zeros(self.num_objectives)
        if self.use_more:
            self.more_weights = 0.5*np.ones(self.num_objectives)
        return 0#, self.cum_reward

    def step(self, action):
        # see figure 1 in Rolf, M. The Need for MORE: Need Systems as Non-Linear Multi-Objective Reinforcement Learning.
        # doi:10.1109/ICDL-EpiRob48136.2020.9278062.
        if action == 1: # move to the right (increase state index)
            self.state = min(self.state + 1, self.num_observations - 1)
        else:
            self.state = max(self.state - 1, 0) # move to the left (decrease state index)
        # determine rewards
        food = 0
        water = 0
        if self.state == 0: # first state yields more food, but decreases water
            food = 0.1
            water = -0.09
        elif self.state == self.num_observations - 1: # last state yields more water, but decreases food
            food = -0.018
            water = 0.02
        else: # all states between the two sources 
            food = -0.001
            water = -0.001

        reward = np.array([food, water])

        # update dynamic weight and renormalize
        if self.use_more:
            self.more_weights = self.weights**self.gamma_past * np.exp(-reward)
            self.more_weights /= self.weights.sum()

        # update cumulative reward
        #self.cum_reward += self.gamma**self.step_counter*reward
        #self.cum_reward = self.gamma_past*self.cum_reward + reward

        terminal = False
        self.step_counter += 1
        if self.step_counter >= self.max_steps:
            terminal = True

        #return (self.state, self.cum_reward), reward, terminal, {}
        return self.state, reward, terminal, {}

    def _get_obs(self):
        return self.state 
                #{"discrete_state": self.state,
                #"cum_reward": self.cum_reward
                #}
    
    def render(self, mode="human"):
        if mode == 'rgb_array':
            return np.array(self.state)#, *self.cum_reward.tolist()]) # return RGB frame suitable for video
        elif mode == 'human':
            print("-----")
            print(f"step: {self.step_counter}, state: {self.state}")#\ncumulative food: {self.cum_reward[0]}\ncumulative water: {self.cum_reward[1]}")
            print("-----")
        else:
            super(FoodWaterBalance, self).render(mode=mode) # just raise an exception

    def close(self):
        pass

    def seed(self, seed=None):
        self.seed = seed if not seed is None else np.random.randint(2**32) 


if __name__ == '__main__':
    import gym
    import mo_gym
    from gym.spaces.utils import flatdim
    #env = FoodWaterBalance(max_steps=100)
    env = gym.make("food-water-balance-v0", max_steps=100)
    assert flatdim(env.action_space) == 2
    assert flatdim(env.observation_space) == 3

    done = False
    obs = env.reset()
    while True:
        env.render()
        obs, r, done, info = env.step(env.action_space.sample())
        if done:
            break