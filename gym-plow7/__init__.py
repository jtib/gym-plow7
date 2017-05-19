from gym.envs.registration import register

register(
        id='plow7-v0',
        entry_point='gym_plow7.envs:Plow7Env',
        )
#register(
#        id='plow7_extrahard-v0',
#        entry_point='gym_plow7.envs:Plow7ExtraHardEnv',
#        )
