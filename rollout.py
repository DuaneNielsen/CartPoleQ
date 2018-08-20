import gym
from abc import abstractmethod, ABC
from mentality import Observable, OpenCV, ImageVideoWriter, View
import pickle
import models
import torch
import torchvision.transforms.functional as tvf

class Policy(ABC):
    @abstractmethod
    def action(self, observation): raise NotImplementedError

class RandomPolicy():
    def __init__(self, env):
        self.env = env

    def action(self, observation):
        return self.env.action_space.sample()

class Rollout(Observable):
    def __init__(self, env):
        Observable.__init__(self)
        self.env = env

    def rollout(self, policy, max_timesteps=100):
        observation = self.env.reset()
        for t in range(max_timesteps):
            screen = self.env.render(mode='rgb_array')
            self.updateObservers('input', screen, 'numpyRGB')
            action = policy.action(observation)
            self.updateObservers('screen_action',(screen, action),'screen_action')
            observation, reward, done, info = self.env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        self.endObserverSession()

class ActionEmbedding():
    def __init__(self, env):
        self.env = env

    def toTensor(self, action):
        action_t = torch.zeros(self.env.action_space.n)
        action_t[action] = 1.0
        return action_t

"""
update takes a tuple of ( numpyRGB, integer )
and saves as tensors
"""
class ActionEncoder(View):
    def __init__(self, model, env, filename):
        View.__init__(self)
        self.model = model
        self.model.eval()
        self.session = 1
        self.z_list = []
        self.a_list = []
        self.filename = filename
        self.action_embedding = ActionEmbedding(env)
        self.device = torch.device('cpu')

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def update(self, screen_action, format):
        x = tvf.to_tensor(screen_action[0].copy()).unsqueeze(0).to(self.device)
        mu, logsigma = self.model.encode(x)
        a = self.action_embedding.toTensor(screen_action[1]).unsqueeze(0)
        self.z_list.append(mu)
        self.a_list.append(a)

    def endSession(self):

        z = torch.stack(self.z_list, dim=0)
        a = torch.stack(self.a_list, dim=0)

        file = open('data/cart/latent/' + self.filename + str(self.session),'wb')
        pickle.dump((a,z), file=file)
        self.session += 1
        self.z_list = []
        self.a_list = []



if __name__ == '__main__':

    device = torch.device("cuda")
    env = gym.make('CartPole-v0')
    random_policy = RandomPolicy(env)
    rollout = Rollout(env)
    #rollout.registerObserver('input', OpenCV('input'))
    #rollout.registerObserver('input', ImageVideoWriter('data/video/cart/','cartpole'))
    cvae = models.ConvVAE.load('conv_run2_cart')
    ae = ActionEncoder(cvae, env, 'run').to(device)
    rollout.registerObserver('screen_action', ae)


    for i_episode in range(20):
        rollout.rollout(random_policy)
