import gym
for env in gym.envs.registry.all():
    if 'Lunar' in env.id:
        print(env.id)

import torch
print(torch.__version__)