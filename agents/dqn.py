import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections

from agents.agent_base import Agent
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

class DQN(Agent):
    def __init__(self, state_shape, num_actions, config, checkpoint_dir=None):
        super().__init__(config)
        self.num_actions = num_actions

        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon = self.epsilon_start

        self.online_network = DQNNetwork(state_shape, num_actions)

        self.target_network = DQNNetwork(state_shape, num_actions)
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
            self.epsilon = training_state["epsilon"]

        self.online_network.to(self.device)
        self.target_network.to(self.device)
        self.target_network.eval()

        self.optimizer = optim.Adam(self.online_network.parameters(), config["learning_rate"])
        if checkpoint_dir is not None:
            self.optimizer.load_state_dict(training_state["optimizer"])

        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(config["replay_buffer_size"])

    @torch.no_grad()
    def get_action(self, state):
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            self.online_network.eval()

            # cast gym.wrappers.LazyFrames to np.ndarray
            state = torch.FloatTensor(np.asarray(state)).unsqueeze(0).to(self.device)
            action = torch.argmax(self.online_network(state)).item()

        if self.is_training:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
        return action

    def update_online(self):
        transitions = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # cast gym.wrappers.LazyFrames to np.ndarray
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        max_next_qs = torch.max(self.target_network(next_states), dim=-1).values
        targets = rewards + (~dones) * self.gamma**self.n_step_td * max_next_qs

        self.online_network.train()
        predictions = self.online_network(states)[range(states.shape[0]), actions]

        loss = self.criterion(predictions, targets)

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
            "epsilon": self.epsilon
        }, output_dir / "training_state.pt")

class DQNTest:
    def __init__(self):
        self.num_actions = 12
        self.state_shape = (4, 84, 84)
        self.epsilon = 0.1
        self.skip_frames = 4

        self.online_network = DQNNetwork(self.state_shape, self.num_actions)
        self.online_network.load_state_dict(
            torch.load("models/dqn_pt.16000.model.pt", weights_only=False)
        )
        self.online_network.eval()

        self.frame_skipped = 0
        self.action = None
        self.frame_stack = collections.deque(maxlen=self.state_shape[0])

    def act(self, observation):
        if self.frame_skipped == 0:
            observation = util.preprocess_state(self.state_shape[1:], observation)

            if len(self.frame_stack) == 0:
                for _ in range(self.state_shape[0]):
                    self.frame_stack.append(observation)
            else:
                self.frame_stack.append(observation)

            if random.random() < self.epsilon:
                self.action = random.randint(0, self.num_actions - 1)
            else:
                with torch.no_grad():
                    state = torch.FloatTensor(np.array(self.frame_stack)).unsqueeze(0)
                    self.action = torch.argmax(self.online_network(state)).item()

        self.frame_skipped = (self.frame_skipped + 1) % self.skip_frames
        return self.action

class _DQNTest:
    def __init__(self, state_shape, num_actions, model_dir):
        self.num_actions = num_actions
        self.epsilon = 0.1

        self.online_network = DQNNetwork(state_shape, num_actions)
        self.online_network.load_state_dict(
            torch.load(model_dir / "model.pt", weights_only=False)
        )
        self.online_network.eval()

    @torch.no_grad()
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            # cast gym.wrappers.LazyFrames to np.ndarray
            state = torch.FloatTensor(np.asarray(state)).unsqueeze(0)
            return torch.argmax(self.online_network(state)).item()
