import gym
import gym.spaces
import numpy as np
import time


env = gym.make('FrozenLake-v0')

# init Q table
Q = np.zeros([env.observation_space.n, env.action_space.n])

#hyper params
lr = 0.8
y = 0.95 #discount factor
num_episodes = 5000
rList = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False
    j = 0
    while j < 99:
        j+=1

        action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n) * (1./(i+1)))

        next_state, reward, done, _ = env.step(action)

        Q[state,action] = Q[state,action] + lr * (reward + np.max(Q[next_state,:]) - Q[state,action])
        rAll += reward
        state = next_state
        if done:
            break
    rList.append(rAll)

done = False
state = env.reset()
while not done:
    action = np.argmax(Q[state, :])
    print(action)
    next_state, reward, done, _ = env.step(action)
    env.render()
    time.sleep(0.2)
    state = next_state
