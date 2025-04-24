import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from replay_buffer import ReplayBuffer
import util


class DQNNetwork(nn.Module):
    def __init__(self, state_shape, num_actions):
        super().__init__()

        channel, height, width = state_shape
        height, width = util.shape_after_conv2d(height, width, kernel_size=(8, 8), stride=(4, 4))
        height, width = util.shape_after_conv2d(height, width, kernel_size=(4, 4), stride=(2, 2))
        height, width = util.shape_after_conv2d(height, width, kernel_size=(3, 3), stride=(1, 1))

        # similar network architecture as in rainbow dqn paper
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * height * width, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQN:
    def __init__(self, state_shape, num_actions, config):
        self.online_network = DQNNetwork(state_shape, num_actions)

        self.target_network = DQNNetwork(state_shape, num_actions)
        for param in self.target_network.parameters():
            param.requires_grad = False
        # shallow copy, will not have required_grad=True
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.optimizer = optim.Adam(self.online_network.parameters(), config["learning_rate"])
        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(config["replay_buffer_size"])

        self.num_actions = num_actions
        self.batch_size = config["batch_size"]
        self.device = config["device"]
        self.gamma = config["gamma"]

        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon = self.epsilon_start

        self.online_network = self.online_network.to(self.device)
        self.target_network = self.target_network.to(self.device)
        self.target_network.eval()

    @torch.no_grad()
    def get_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            self.online_network.eval()

            # cast gym.wrappers.LazyFrames to np.ndarray
            state = torch.FloatTensor(np.asarray(state)).unsqueeze(0).to(self.device)
            action = torch.argmax(self.online_network(state)).item()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
        return action

    def update_target(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def update_online(self):
        transitions = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # cast gym.wrappers.LazyFrames to np.ndarray
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        max_next_qs = torch.max(self.target_network(next_states), dim=-1).values
        targets = rewards + (~dones) * self.gamma * max_next_qs

        self.online_network.train()
        predictions = self.online_network(states)[range(states.shape[0]), actions]

        loss = self.criterion(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def on_episode_end(self, episode):
        # self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
        pass

    def save(self, output_dir):
        self.online_network = self.online_network.cpu()
        torch.save(self.online_network.state_dict(), output_dir / f"model.pt")
        self.online_network = self.online_network.to(self.device)

        torch.save({
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.buffer,
        }, output_dir / f"training_state.pt")
