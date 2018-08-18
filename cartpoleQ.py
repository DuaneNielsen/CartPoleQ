import gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import models
import mentality
from mentality import OpenCV, ImageFileWriter, SummaryWriterWithGlobal


env = gym.make('CartPole-v0').unwrapped

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Eyes(mentality.Observable):
    def __init__(self, env, vae):
        super(Eyes, self).__init__()
        self.env = env
        self.resize = T.Compose([T.ToPILImage(),
                       T.Resize((32,48), interpolation=Image.CUBIC),
                       T.ToTensor()])
        self.transF = T.Compose([T.ToPILImage(),
                              T.ToTensor()])
        self.pilF = T.Compose([T.ToPILImage()])
        self.vae = vae

    def screen_width(self, screen):
        return screen.shape[2]

    def get_cart_location(self, screen):
        sw = self.screen_width(screen)
        world_width = env.x_threshold * 2
        scale = self.screen_width(screen) / world_width
        return int(self.env.state[0] * scale + sw / 2.0,)

    def fill_screen(self, screen, view_width):
        patch_shape = (screen.shape[0], screen.shape[1], view_width //2)
        patch = torch.zeros(patch_shape)
        screen = torch.cat((patch, screen, patch), dim=2)
        return screen

    def get_screen(self):
        screen = self.env.render(mode='rgb_array')

        self.updateObservers('raw',screen,'numpyRGB')

        screen = self.transF(screen)

        #chop off above and below
        screen=screen[:,160:640,:]

        #drawPILTensor(screen)

        #center the a smaller view around the cart
        #cart_location = self.get_cart_location(screen)
        #view_width = 320
        #half_view_width = view_width // 2

        #fill the left and right edges of the screen with black space
        #screen = self.fill_screen(screen, view_width)
        #cart_location = cart_location + half_view_width

        #take a slice around the cart
        #left = cart_location - half_view_width
        #right = cart_location + half_view_width
        #screen = screen[:,:,left:right]

        screen = self.resize(screen).to(device)

        self.updateObservers('input', screen, 'tensorPIL')

        screen_recon, _, _ = self.vae(screen)

        self.updateObservers('recon', screen_recon.detach().view(1, 3, 32, 48), 'tensorPIL')

        #move the tensor to the processing memory
        return screen.unsqueeze(0).to(device)

    def render(self):
        screen = self.get_screen().cpu()
        image = screen.squeeze().permute(1, 2, 0).numpy()
        self.showImage(image)


env.reset()
plt.figure()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

IMAGE_SIZE = 3 * 32 * 48

tb = SummaryWriterWithGlobal('cartpole')

policy_net = models.DQN().to(device)
target_net = models.DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

vae = models.ThreeLayerLinearVAE(IMAGE_SIZE, 10).to(device)
vae_optim = optim.Adam(vae.parameters(), lr=1e-3)

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


eye = Eyes(env, vae)
#eye.registerObserver('raw', OpenCV('raw'))
eye.registerObserver('raw', OpenCV('raw'))
eye.registerObserver('recon', OpenCV('recon'))
eye.registerObserver('raw', ImageFileWriter('data/images/fullscreen','raw'))
debug_observer = OpenCV('debug')
#eye.registerObserver("raw", Plotter(1))
#eye.registerObserver('input', Plotter(3))

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_theshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_theshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype = torch.long)

episode_durations = []





def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title("Training...")
    plt.ylabel("Duration")
    plt.xlabel("Episode")
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())

    plt.pause(0.001)

# Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy_with_logits(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    tb.tensorboard_scaler('loss/KLD', KLD)
    tb.tensorboard_scaler('loss/BCE', BCE)
    return BCE + KLD

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device = device, dtype=torch.uint8)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #VAE
    tb.tensorboard_step()
    debug_observer.update(state_batch[1])
    recon_x, mu, logvar = vae(state_batch)
    loss = vae_loss_function(recon_x, state_batch, mu, logvar)
    tb.tensorboard_scaler('vae_loss', loss)
    vae_optim.zero_grad()
    loss.backward()
    vae_optim.step()


    #Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    #Compute V(s_t+1) for all next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    #Compute expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    #Huber LOSS
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))


    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp(-1, 1)
    optimizer.step()

# main loop

num_episodes = 5000
for i_episode in range(num_episodes):
    env.reset()
    last_screen = eye.get_screen()
    current_screen = eye.get_screen()
    #state = current_screen - last_screen\
    state = current_screen
    for t in count():
        # select and perform action
        action = select_action(state)
        #print(action)
        _ , reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = eye.get_screen()
        if not done:
            #next_state = current_screen - last_screen
            next_state = current_screen
        else:
            next_state = None

        #Store the transition in replay memory
        memory.push(state, action, next_state, reward)

        state = next_state

        #perform one step of optimization on the target network
        optimize_model()
        if done:
            #episode_durations.append(t + 1)
            tb.tensorboard_scaler('duration_till_done', t + 1)
            #plot_durations(t+1)
            break

    #update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.show()

