import gym
from agents.dpdn import DPDNTest

# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    def __init__(self):
        self.agent = DPDNTest()

    def act(self, observation):
        return self.agent.act(observation)
