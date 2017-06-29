import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_argos3.envs.argos3_env import Argos3Env, logger

import numpy as np

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
        logger.info("Env made")

    def setParams(self, number_footbots, min_speed=2, max_speed=25, dt="numerical"):
        self.nbFb = number_footbots
        self.obs_len = number_footbots * (1 + 2*24 + 1)
        self.observation_space = spaces.Box(-np.ones([self.obs_len]), np.ones([self.obs_len])) # will have to normalize observations
        self.av_speeds = np.zeros(number_footbots)
        self.data_type = dt
        super().setParams(number_footbots, min_speed, max_speed, dt)

    def process_raw_state(self, raw_state):
        logger.debug(f"Proximity sensor = {str(raw_state[:48*self.nbFb])}")
        logger.debug(f"Footbot speed = {str(raw_state[48*self.nbFb:49*self.nbFb])}")
        logger.debug(f"Distance to departure = {str(raw_state[49*self.nbFb:])}")
        logger.debug(f"Collisions detected = {sum(np.array(raw_state[:48*self.nbFb])>.95)}")

        self.all_speeds[self.t] = raw_state[48*self.nbFb:49*self.nbFb]

        return raw_state

    def _reset(self):
        self.all_speeds = np.zeros((self.t_max, self.nbFb))
        self.t = 0
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
        logger.debug(f'proximities shape = {proximities.shape}')
        prox = np.sum(proximities, axis=1)
        self.av_speeds = (self.av_speeds*self.t + speeds)/(self.t+1)

        done = all(dist_dep > 7) # done is good
        reward = sum(self.av_speeds - 5*sum(prox[:,0]>.95) + 5*dist_dep) # might need to normalize these; speed good, collisions bad
        if done:
            reward += 100

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
        env.step([.5])

        if (i+1)%5 == 0:
            env.reset()

if __name__ == '__main__':
    test_plow7_env()
