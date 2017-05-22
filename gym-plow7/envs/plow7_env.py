import gym
from gym import spaces

from gym_argos3.argos3_env import Argos3Env, logger

import numpy as np

class Plow7Env(Argos3Env):
    """
    OpenAI gym environment for a specific crossroads setting in ARGoS3.
    action = [throttle]
    No steering wheel.
    """
    def __init__(self):
        super.__init__(width=128, height=128, batchmode=False)
        self.t_max = 300 #30s at 10fps
        self.t0 = 0
        obs_dim = 39 # TODO: understand and correct that
        self.observation_space = spaces.Box(-np.ones([obs_dim]), np.ones([obs_dim]))

    def process_raw_state(self, raw_state):
        logger.debug(f"Footbot speed = {str(raw_state[:8])}")
        logger.debug(f"Distance to departure = {str(raw_state[8:16])}")
        logger.debug(f"Proximity sensor = {str(raw_state[16:])}")
        logger.debug(f"Collisions detected = {sum(np.array(raw_state[16:])>.8)}")

        self.all_speeds[self.t] = raw_state[:8]
        av_speeds = np.mean(self.all_speeds, 0)

        return np.concatenate(raw_state[:], av_speeds)

    def _reset(self):
        self.speed = 0
        # learning everything at the same time
        # not realistic (compared to real life), but easier to program
        # not elegant either
        self.t = 0

    def _step(self, action):
        state = self.receive()
        state = self.process_raw_state(state)

        position = state[0] # all car positions
        pos_diff = state[1] # distance from init. pos
        speed = state[2] # all car speeds
        av_speed = state[3] # average speed (of all cars/each?)
        proximities = state[4] # proximity sensor readings

        done = self.t > self.t_max or all(np.abs(position - pos_diff) > 7) # done is good
        reward = av_speed - sum(proximities>0.8) # might need to normalize av_speed; speed good, collisions bad
        if done:
            reward += 10

        self.t += 1

        return reward, done

    def _render(self, mode='human', close=False):
        pass
