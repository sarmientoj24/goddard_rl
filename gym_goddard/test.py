import gym

from gym import ActionWrapper
import numpy as np

class NormalizedActions(ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action


if __name__ == '__main__':
    env = gym.make('gym_goddard:Goddard-v0')
    new_env = NormalizedActions(env)
    print(new_env.action_space, new_env.action_space.high, new_env.action_space.low)