import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import collections

from noisy_network import NoisyLinear
from agents.agent_base import Agent
from replay_buffer import PrioritizedReplayBuffer
import util


class RainbowNetwork(nn.Module):
    def __init__(self, state_shape, num_actions, z_support, sigma0):
        super().__init__()
        self.num_actions = num_actions
        self.z_support = z_support
        self.num_atoms = self.z_support.shape[0]

        channel, height, width = state_shape
        height, width = util.shape_after_conv2d(height, width, kernel_size=(8, 8), stride=(4, 4))
        height, width = util.shape_after_conv2d(height, width, kernel_size=(4, 4), stride=(2, 2))
        height, width = util.shape_after_conv2d(height, width, kernel_size=(3, 3), stride=(1, 1))

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
            NoisyLinear(512, 1 * self.num_atoms, sigma0)
        )

        self.advantage = nn.Sequential(
            NoisyLinear(64 * height * width, 512, sigma0),
            nn.ReLU(),
            NoisyLinear(512, num_actions * self.num_atoms, sigma0)
        )

    def q_distribution(self, x):
        x = self.convolution(x)

        value = self.value(x).view(-1, 1, self.num_atoms)
        advantage = self.advantage(x).view(-1, self.num_actions, self.num_atoms)

        q_logit = value + advantage - advantage.mean(dim=-2, keepdim=True)

        # input shape (batch_size, channel, width, height)
        # return shape (batch_size, num_actions, num_atoms)
        return F.softmax(q_logit, dim=-1)

    def q_value(self, x):
        # input shape (batch_size, channel, width, height)
        # return shape (batch_size, num_actions)
        return (self.q_distribution(x) * self.z_support).sum(dim=-1)

    def resample_noise(self):
        self.value[0].resample_noise()
        self.value[2].resample_noise()
        self.advantage[0].resample_noise()
        self.advantage[2].resample_noise()

    def zero_noise(self):
        self.value[0].zero_noise()
        self.value[2].zero_noise()
        self.advantage[0].zero_noise()
        self.advantage[2].zero_noise()

class Rainbow(Agent):
    def __init__(self, state_shape, num_actions, config, checkpoint_dir=None):
        super().__init__(config)
        self.num_atoms = config["num_atoms"]
        self.v_min = config["v_min"]
        self.v_max = config["v_max"]

        self.z_support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        self.idx_offset = (torch.arange(self.batch_size).unsqueeze(1) * self.num_atoms)\
                          .expand(self.batch_size, self.num_atoms)
        assert config["steps_before_training"] + config["steps_per_online_update"]\
               >= config["n_step_td"]-1 + self.batch_size,\
               "Current implementation requires collecting more transitions than batch size "\
               "before updating the online network"


        self.online_network = RainbowNetwork(
            state_shape, num_actions, self.z_support, config["sigma0"]
        )

        ### load pretrained model ###
        from agents.dqn import DQNNetwork

        pretrained = DQNNetwork(state_shape, num_actions)
        pretrained.load_state_dict(
            torch.load("[file path].pt", weights_only=False)
        )

        for i in 0, 2, 4:
            self.online_network.convolution[i].load_state_dict(pretrained.net[i].state_dict())
        #############################

        self.target_network = RainbowNetwork(
            state_shape, num_actions, self.z_support, config["sigma0"]
        )
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
        self.z_support = self.z_support.to(self.device)
        self.idx_offset = self.idx_offset.to(self.device)
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.online_network.parameters(), config["learning_rate"], eps=config["adam_epsilon"]
        )
        if checkpoint_dir is not None:
            self.optimizer.load_state_dict(training_state["optimizer"])

        self.replay_buffer = PrioritizedReplayBuffer(config["replay_buffer_size"], config["alpha"])

        self.beta0 = config["beta0"]
        self.total_episodes = config["resume_from_checkpoint"] + config["total_episodes"]
        self.on_episode_end(config["resume_from_checkpoint"])  # set up self.beta


        self.output_history = torch.zeros(self.num_atoms).to(self.device)
        self.output_count = 0
        self.bellman = torch.zeros(self.num_atoms).to(self.device)
        self.update_count = 0

    @torch.no_grad()
    def get_action(self, state):
        self.online_network.eval()

        # cast gym.wrappers.LazyFrames to np.ndarray
        state = torch.FloatTensor(np.asarray(state)).unsqueeze(0).to(self.device)

        self.online_network.resample_noise()
        distribution = self.online_network.q_distribution(state)  # shape (1, num_actions, num_atoms)
        q = (distribution * self.z_support).sum(dim=-1)  # shape (1, num_actions)
        action = torch.argmax(q).item()

        self.output_history += distribution[0, action]
        self.output_count += 1

        return action

    def update_online(self):
        transitions, weights, indices = self.replay_buffer.sample(self.batch_size, self.beta)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # cast gym.wrappers.LazyFrames to np.ndarray
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        self.online_network.eval()
        with torch.no_grad():
            self.online_network.resample_noise()
            q = self.online_network.q_value(next_states)  # (batch_size, num_actions)
            online_next_actions = torch.argmax(q, dim=-1)  # (batch_size,)

        self.target_network.resample_noise()
        target_next_dists = self.target_network.q_distribution(next_states)[
            range(self.batch_size), online_next_actions, :
        ]  # (batch_size, num_atoms)

        bellman_update = (
            rewards.unsqueeze(1) + (~dones.unsqueeze(1)) * self.gamma_n * self.z_support
        )

        self.bellman += bellman_update.mean(dim=0)
        self.update_count += 1

        bellman_update = bellman_update.clamp(self.v_min, self.v_max)  # (batch_size, num_atoms)
        b = (bellman_update - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        m = torch.zeros(self.batch_size, self.num_atoms).to(self.device)
        m.view(-1).index_add_(
            0, (self.idx_offset + l).view(-1), (target_next_dists * (u - b)).view(-1)
        )
        m.view(-1).index_add_(
            0, (self.idx_offset + u).view(-1), (target_next_dists * (b - l)).view(-1)
        )

        self.online_network.train()
        self.online_network.resample_noise()
        predictions = self.online_network.q_distribution(states)[
            range(self.batch_size), actions, :
        ]

        loss = - (m * predictions.log()).sum(dim=-1)  # (batch_size,)

        priorities = (loss + 1e-6).detach().cpu().tolist()
        self.replay_buffer.update_priorities(indices, priorities)

        loss = (weights * loss).mean()

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
        # linearly anneal beta to 1
        self.beta = self.beta0 + episode/(self.total_episodes - 1) * (1 - self.beta0)

    def on_log(self, logger):
        k = 5

        sort, indices = (self.output_history / self.output_count).sort()
        sort = sort.cpu().numpy()
        indices = indices.cpu().numpy()
        logger.debug("output distribution:\n"
                     f"\tlargest {sort[-k:]}, indices {indices[-k:]}\n"
                     f"\tsmallest {sort[:k]}, indices {indices[:k]}"
        )
        self.output_history.zero_()
        self.output_count = 0

        bellman = (self.bellman / self.update_count).cpu().numpy()
        logger.debug("bellman update:\n"
                     f"\tlargest {bellman[-2*k :]}\n"
                     f"\tsmallest {bellman[: 2*k]}"
        )
        self.bellman.zero_()
        self.update_count = 0

class RainbowTest:
    def __init__(self):
        num_actions = 12
        self.state_shape = (4, 84, 84)
        self.skip_frames = 4
        sigma0 = 0.5
        z_support = torch.linspace(-20, 300, 51)

        self.online_network = RainbowNetwork(self.state_shape, num_actions, z_support, sigma0)
        self.online_network.load_state_dict(
            torch.load("models/rainbow/checkpoint-10000/model.pt", weights_only=False)
        )
        self.online_network.eval()
        self.online_network.zero_noise()

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

            with torch.no_grad():
                state = torch.FloatTensor(np.array(self.frame_stack)).unsqueeze(0)

                self.online_network.resample_noise()
                self.action = torch.argmax(self.online_network.q_value(state)).item()

        self.frame_skipped = (self.frame_skipped + 1) % self.skip_frames
        return self.action
