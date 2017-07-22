import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_argos3.envs.argos3_env import Argos3Env, logger

import numpy as np

def getUpdatedReward(reward, multiplier):
    ret = reward
    if ret >= 0:
        ret *= multiplier
    else:
        ret *= -multiplier
    return ret

class Plow7Env(Argos3Env):
    """
    OpenAI gym environment for a specific crossroads setting in ARGoS3.
    action = [throttle]
    No steering, the footbots move in a straight line.
    """
    def __init__(self):
        super().__init__(width=128, height=128)
        self.t_max = 3600 * 10 #3600s at 10fps
        self.t0 = 0
        self.t = 0
        logger.info("Env made")

    def setParams(self, number_footbots, min_speed=2, max_speed=25, dt="numerical"):
        self.nbFb = number_footbots
        self.obs_len = number_footbots * (1 + 2*24 + 1)
        self.observation_space = spaces.Box(-np.ones([self.obs_len]), np.ones([self.obs_len])) # will have to normalize observations
        self.av_speeds = np.zeros(number_footbots)
        self.data_type = dt
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.timesteps = 0
        super().setParams(number_footbots, min_speed, max_speed, dt)

    def process_raw_state(self, raw_state):
        return raw_state

    def _reset(self):
        """ Reconnects to the simulator if necessary.
        This does not reset the state of the simulator.
        This is for online learning.
        """
        state, frame = super()._reset()
        state = self.process_raw_state(state)
        return state

    def _step(self, action):
        action = self.action_space.low + (action+np.ones(self.nbFb))*(self.action_space.high[0]-self.action_space.low[0])/2
        self.send(action)
        state, frame = self.receive()
        state = self.process_raw_state(state)

        speeds = state[48*self.nbFb:49*self.nbFb] # all fb positions
        print(f"Speeds: {speeds}")
        dist_dep = state[49*self.nbFb:] # distance from init. pos
        proximities = state[:48*self.nbFb] # all fb proxim. readings
        proximities = np.array(proximities)
        proximities = proximities.reshape((self.nbFb,24,2)) # which fb, which sensor, value/angle
        if self.t % 50 == 0:
            print(f"Episode: {self.t}  Speeds: {speeds}")
            print(f"Distances: {dist_dep}")
            print(f"Proximities: {proximities}")
        self.av_speeds = (self.av_speeds*self.t + speeds)/(self.t+1)

        # Dentinger & team:
        # better to use the "distance to collision"
        # rather than plain distance
        # because the two footbots that go in the
        # same direction will stay close to each other
        # but that shouldn't cause a loss in reward.
        #
        # Other improvement: data crunching
        # As it was: data of ALL footbots crushed at once
        # Much better information by taking
        # a reward per action per footbot
        # rather than for all footbots together.

        reward = 1

        for fbProx in proximities:
            for prox in fbProx:
                reward -= 10 * prox[0]**2 # The closer you get, the less I reward you

        for speed in speeds:
            if speed < self.min_speed or speed > self.max_speed:
                reward -= 5
            else:
                # Encouraged to go as fast as possible
                reward += 3 * (speed - self.min_speed) / (self.max_speed - self.min_speed)

        self.timesteps += 1
        done = (self.timesteps >= 300)

        # done = all(dist_dep > 7) # This is broken ; it always returns True after a while
                                   # Not sure how it is calcualted.
        # if done:
        #     reward += 1000
        #     print("DONE")

        self.t += 1
        print("reward: ", reward)
        return state, reward, done, {}

    def _render(self, mode='human', close=False):
        pass


def test_plow7_env():
    import logging
    import gym_argos3
    import gym_plow7
    logger.setLevel(logging.DEBUG)

    env = gym.make('plow7-v0')
    env.unwrapped.conf(loglevel='debug')
    env.reset()
    for i in range(10):
        print(i)
        env.step([.5])

        if (i+1)%5 == 0:
            env.reset()

if __name__ == '__main__':
    test_plow7_env()
