import gym
from abc import abstractmethod, ABC
from mentality import Dispatcher, Observable, OpenCV, ImageVideoWriter, View, ImageFileWriter, Storeable
import pickle
import models
import torch
import torchvision.transforms.functional as tvf
import numpy as np


class Policy(ABC):
    @abstractmethod
    def action(self, observation): raise NotImplementedError


class RandomPolicy():
    def __init__(self, env):
        self.env = env

    def action(self, observation):
        return self.env.action_space.sample()


class Rollout(Dispatcher, Observable):
    def __init__(self, env):
        Dispatcher.__init__(self)
        self.env = env

    def rollout(self, policy, max_timesteps=100):
        observation = self.env.reset()
        for t in range(max_timesteps):
            screen = self.env.render(mode='rgb_array')
            self.updateObserversWithImage('input', screen, 'numpyRGB')
            action = policy.action(observation)
            self.updateObservers('screen_action',(screen, action),{'func':'screen_action'})
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

class ObservationAction:
    def __init__(self):
        self.observation = None
        self.action = None

"""
update takes a tuple of ( numpyRGB, integer )
and saves as tensors
"""
class ActionEncoder(View):
    def __init__(self, model, env, filename, datadir):
        View.__init__(self)
        self.model = model
        self.model.eval()
        self.session = 1
        self.sess_obs_act = None
        self.filename = filename
        self.action_embedding = ActionEmbedding(env)
        self.device = torch.device('cpu')
        self.datadir = datadir
        if datadir is None:
            self.datadir = 'c:\data'
        self.env = env
        self.oa = None

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def update(self, screen_action, metadata):
        with torch.no_grad():
            self.model.eval()
        x = tvf.to_tensor(screen_action[0].copy()).detach().unsqueeze(0).to(self.device)
        mu, logsigma = self.model.encoder(x)
        a = self.action_embedding.toTensor(screen_action[1]).detach()

        obs_n = mu.detach().cpu().numpy()
        act_n = a.cpu().numpy()
        act_n = np.expand_dims(act_n, axis=0)

        if self.oa is None:
            self.oa = ObservationAction()
            self.oa.observation = np.empty(obs_n.shape, dtype='float32')
            self.oa.action = np.empty(act_n.shape, dtype='float32')

        self.oa.observation = np.append(self.oa.observation, obs_n, axis=0)
        self.oa.action = np.append(self.oa.action, act_n, axis=0)

    def endSession(self):

        #todo add code to make directory
        file = open(self.datadir +'/' +  self.env.spec.id + '/latent/' + self.filename + str(self.session),'wb')
        pickle.dump(self.oa, file=file)
        self.session += 1
        self.sess_obs_act = None



if __name__ == '__main__':

    device = torch.device("cuda")
    env = gym.make('SpaceInvaders-v4')
    random_policy = RandomPolicy(env)
    rollout = Rollout(env)
    rollout.registerView('input', OpenCV('input'))
    #rollout.registerObserver('input', ImageVideoWriter('data/video/spaceinvaders/','random'))
    #rollout.registerObserver('input', ImageFileWriter('data/images/spaceinvaders/fullscreen', 'input', 16384))
    #cvae = models.ConvVAE.load('conv_run2_cart')

    name = 'atari_v3'
    #atari_conv = models.AtariConv_v4()
    atari_conv = Storeable.load(r'jenkins-View-10\23b450a23bfbe218aa5070220afdab0db0d58216\AtariConv_v6\([64, 64, 64, 64, 64],)',
                                data_dir='c:/data')
    atari_conv = atari_conv.eval()
    ae = ActionEncoder(atari_conv, env, 'run', 'c:/data').to(device)
    rollout.registerView('screen_action', ae)


    for i_episode in range(100):
        rollout.rollout(random_policy, max_timesteps=1000)
