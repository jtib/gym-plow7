import gym
from gym import error, spaces, utils
from gym.utils import seeding

class Plow7Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.t_max = 300 #30s at 10fps

    def send(self, action, reset=False):


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

    def _reset(self):
        self.speed = 0
        self.t = 0


    def _render(self, mode='human', close=False):
        pass
