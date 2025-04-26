import random
import collections
import numpy as np

from segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer =collections.deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        # np.uint8 is deprecated
        self.buffer.append((state, np.int8(action), reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.size = 0

        self.buffer = [None for _ in range(self.capacity)]
        self.sum_tree = SumSegmentTree(self.capacity)
        self.min_tree = MinSegmentTree(self.capacity)

        self.replace_index = 0
        self.max_priority = 1
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        # np.uint8 is deprecated
        self.buffer[self.replace_index] = (state, np.int8(action), reward, next_state, done)
        self.sum_tree[self.replace_index] = self.max_priority ** self.alpha
        self.min_tree[self.replace_index] = self.max_priority ** self.alpha

        self.size = max(self.size, self.replace_index + 1)
        self.replace_index = (self.replace_index + 1) % self.capacity

    def sample(self, batch_size, beta):
        batch_size = min(batch_size, self.size)

        indices = []
        delta = self.sum_tree.query() / batch_size
        weights = []
        max_weight = (self.size * self.min_tree.query() / self.sum_tree.query()) ** -beta

        for i in range(batch_size):
            greater_than = random.uniform(delta * i, delta * (i + 1))
            index = self.sum_tree.min_index_greater_than(greater_than)
            indices.append(index)

            weight = ((self.size * self.sum_tree[index] / self.sum_tree.query()) ** -beta) / max_weight
            weights.append(weight)

        return [self.buffer[i] for i in indices], weights, indices

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            self.sum_tree[index] = priority ** self.alpha
            self.min_tree[index] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.size


class NStepBuffer:
    def __init__(self, n_step, gamma, buffer):
        self.n_step = n_step
        self.buffer = buffer

        if self.n_step > 1:
            self.gamma = gamma
            self.n_buffer = collections.deque(maxlen=self.n_step)

    def add(self, state, action, reward, next_state, done):
        if self.n_step == 1:
            self.buffer.add(state, action, reward, next_state, done)
            return

        self.n_buffer.append((state, action, reward, next_state, done))

        if done:
            transitions = []

            n_reward = 0
            for transition in reversed(self.n_buffer):
                state, action, reward = transition[:3]
                n_reward = reward + self.gamma * n_reward
                transitions.append((state, action, n_reward, next_state, done))

            for transition in reversed(transitions):
                self.buffer.add(*transition)

            self.n_buffer.clear()
            return

        if len(self.n_buffer) < self.n_step:
            return

        n_reward = 0
        for transition in reversed(self.n_buffer):
            state, action, reward = transition[:3]
            n_reward = reward + self.gamma * n_reward

        self.buffer.add(state, action, n_reward, next_state, done)
