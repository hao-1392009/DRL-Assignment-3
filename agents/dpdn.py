import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections

from noisy_netork import NoisyLinear
from agents.agent_base import Agent
from replay_buffer import PrioritizedReplayBuffer
import util


class DPDNNetwork(nn.Module):
    def __init__(self, state_shape, num_actions, sigma0):
        super().__init__()

        channel, height, width = state_shape
        height, width = util.shape_after_conv2d(height, width, kernel_size=(8, 8), stride=(4, 4))
        height, width = util.shape_after_conv2d(height, width, kernel_size=(4, 4), stride=(2, 2))
        height, width = util.shape_after_conv2d(height, width, kernel_size=(3, 3), stride=(1, 1))

        # similar network architecture as in rainbow dqn paper
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.value = nn.Sequential(
            NoisyLinear(64 * height * width, 512, sigma0),
            nn.ReLU(),
            NoisyLinear(512, 1, sigma0)
        )

        self.advantage = nn.Sequential(
            NoisyLinear(64 * height * width, 512, sigma0),
            nn.ReLU(),
            NoisyLinear(512, num_actions, sigma0)
        )

    def forward(self, x):
        x = self.convolution(x)

        value = self.value(x)
        advantage = self.advantage(x)

        return value + advantage - advantage.mean(-1, keepdim=True)

    def resample_noise(self):
        self.value[0].resample_noise()
        self.value[2].resample_noise()
        self.advantage[0].resample_noise()
        self.advantage[2].resample_noise()

# Double + Prioritized replay buffer + Dueling + Noisy net DQN
class DPDN(Agent):
    def __init__(self, state_shape, num_actions, config, checkpoint_dir=None):
        super().__init__(config)

        self.online_network = DPDNNetwork(state_shape, num_actions, config["sigma0"])

        ### load pretrained model ###
        from agents.dqn import DQNNetwork

        pretrained = DQNNetwork(state_shape, num_actions)
        pretrained.load_state_dict(
            torch.load("[file path].pt", weights_only=False)
        )

        for i in 0, 2, 4:
            self.online_network.convolution[i].load_state_dict(pretrained.net[i].state_dict())

        self.online_network.advantage[0].w_mu.data.copy_(pretrained.net[7].weight.data)
        self.online_network.advantage[0].b_mu.data.copy_(pretrained.net[7].bias.data)
        self.online_network.advantage[2].w_mu.data.copy_(pretrained.net[9].weight.data)
        self.online_network.advantage[2].b_mu.data.copy_(pretrained.net[9].bias.data)
        #############################

        self.target_network = DPDNNetwork(state_shape, num_actions, config["sigma0"])
        for param in self.target_network.parameters():
            param.requires_grad = False

        if checkpoint_dir is None:
            # shallow copy, will not have required_grad=True
            self.target_network.load_state_dict(self.online_network.state_dict())
        else:
            self.online_network.load_state_dict(
                torch.load(checkpoint_dir / "model.pt", weights_only=False, map_location=self.device)
            )

            training_state = torch.load(checkpoint_dir / "training_state.pt", weights_only=False)
            self.target_network.load_state_dict(training_state["target_network"])

        self.online_network.to(self.device)
        self.target_network.to(self.device)
        self.target_network.eval()

        self.optimizer = optim.Adam(self.online_network.parameters(), config["learning_rate"])
        if checkpoint_dir is not None:
            self.optimizer.load_state_dict(training_state["optimizer"])

        self.replay_buffer = PrioritizedReplayBuffer(config["replay_buffer_size"], config["alpha"])

        self.beta0 = config["beta0"]
        self.total_episodes = config["resume_from_checkpoint"] + config["total_episodes"]
        self.on_episode_end(config["resume_from_checkpoint"])  # set up self.beta

    @torch.no_grad()
    def get_action(self, state):
        self.online_network.eval()

        # cast gym.wrappers.LazyFrames to np.ndarray
        state = torch.FloatTensor(np.asarray(state)).unsqueeze(0).to(self.device)

        self.online_network.resample_noise()
        return torch.argmax(self.online_network(state)).item()

    def update_online(self):
        transitions, weights, indices  = self.replay_buffer.sample(self.batch_size, self.beta)
        states, actions, rewards, next_states, dones = zip(*transitions)
        batch_size = len(transitions)

        # cast gym.wrappers.LazyFrames to np.ndarray
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        self.online_network.eval()
        with torch.no_grad():
            self.online_network.resample_noise()
            online_next_actions = torch.argmax(self.online_network(next_states), dim=-1)

        self.target_network.resample_noise()
        target_next_qs = self.target_network(next_states)[range(batch_size), online_next_actions]
        targets = rewards + (~dones) * self.gamma * target_next_qs

        self.online_network.train()
        self.online_network.resample_noise()
        predictions = self.online_network(states)[range(batch_size), actions]

        td_error = targets - predictions

        priorities = (td_error.abs() + 1e-6).detach().cpu().tolist()
        self.replay_buffer.update_priorities(indices, priorities)

        loss = (weights * td_error.square()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, output_dir):
        self.online_network.cpu()
        torch.save(self.online_network.state_dict(), output_dir / "model.pt")
        self.online_network.to(self.device)

        torch.save({
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, output_dir / "training_state.pt")

    def on_episode_end(self, episode):
        # linear annealing to 1
        self.beta = self.beta0 + episode/(self.total_episodes - 1) * (1 - self.beta0)
