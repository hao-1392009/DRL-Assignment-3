import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TransformObservation
import numpy as np


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self.skip):
            # repeat the same action
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return state, total_reward, done, info

class ModifiedResizeObservation(ResizeObservation):
    # The same as that in the base class, except that we do not perform unnecessary dimension expansion.
    def observation(self, observation):
        import cv2

        observation = cv2.resize(
            observation, self.shape[::-1], interpolation=cv2.INTER_AREA
        )
        return observation

def preprocess_env(env, frame_size, skip_frames, stack_frames, lz4_compress):
    """
    observations returned by the env are gym.wrappers.LazyFrames
    """
    env = FrameSkip(env, skip_frames)
    env = GrayScaleObservation(env)
    env = ModifiedResizeObservation(env, frame_size)
    env = TransformObservation(env, lambda observation: observation.astype(np.float32) / 255)
    env = FrameStack(env, stack_frames, lz4_compress)
    return env
