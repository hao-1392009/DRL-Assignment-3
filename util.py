import random
import numpy as np
import torch


def fix_random_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

def shape_after_conv2d(height, width, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    return int((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1),\
           int((width  + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

def preprocess_state(frame_size: tuple, observation: np.ndarray):
    import cv2
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    observation = cv2.resize(
        observation, frame_size[::-1], interpolation=cv2.INTER_AREA
    )
    observation = observation.astype(np.float32) / 255
    return observation
