import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_argos3.envs.argos3_env import Argos3Env, logger

import numpy as np

class Plow7Env(Argos3Env):
    """
    OpenAI gym environment for a specific crossroads setting in ARGoS3.
    action = [throttle]
    No steering wheel.
    """
    def __init__(self):
        super().__init__(width=128, height=128, batchmode=False)
        self.t_max = 30 * 10 #30s at 10fps
        self.t0 = 0
        self.obs_len = 4 * 8
        self.observation_space = spaces.Box(-np.ones([self.obs_len]), np.ones([self.obs_len])) # will have to normalize observations

    def process_raw_state(self, raw_state):
        logger.debug(f"Footbot speed = {str(raw_state[:8])}")
        logger.debug(f"Distance to departure = {str(raw_state[8:16])}")
        logger.debug(f"Proximity sensor = {str(raw_state[16:])}")
        logger.debug(f"Collisions detected = {sum(np.array(raw_state[16:])>.8)}")

        self.all_speeds[self.t] = raw_state[:8]
        av_speeds = np.mean(self.all_speeds, 0)

        return np.concatenate(raw_state[:], av_speeds)

    def _reset(self):
        # learning everything at the same time
        # not realistic (compared to real life), but easier to program
        # not elegant either
        self.all_speeds = np.zeros((self.t_max, 8))
        self.t = 0
        state, frame = super()._reset()
        state = self.process_raw_state(state)
        return state

    def _step(self, action):
        self.send(action)
        state, frame = self.receive()
        state = self.process_raw_state(state)

        speeds = state[:8] # all fb positions
        dist_dep = state[8:16] # distance from init. pos
        proximities = state[16:24] # all fb proxim. readings
        av_speeds = state[24:] # average speeds (for each fb)

        done = self.t > self.t_max or all(dist_dep > 7) # done is good
        reward = av_speeds - sum(proximities>0.8) # might need to normalize av_speed; speed good, collisions bad
        if done:
            reward += 10

        self.t += 1

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
        env.step([.0, 1.0])

        if (i+1)%5 == 0:
            env.reset()

if __name__ == '__main__':
    test_plow7_env()
