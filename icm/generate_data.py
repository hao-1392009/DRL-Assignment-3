import argparse
import importlib
import pathlib
import os
import pickle
import sys

import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import tqdm
import numpy as np
import torch
import lz4.block

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env_preprocessor import preprocess_env
import util


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", help="class name of the agent (case-sensitive)")
    parser.add_argument("--frame_size", type=int, nargs="*")
    parser.add_argument("--skip_frames", type=int)
    parser.add_argument("--stack_frames", type=int)
    parser.add_argument("--lz4_compress", action="store_true")
    parser.add_argument("--num_train_data", type=int)
    parser.add_argument("--num_eval_data", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output_dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--random_act_ratio", type=float, help="ratio of data generated with random action selection")
    parser.add_argument("--epsilon", type=float, help="for random action selection")
    parser.add_argument("--random_init_steps", type=int)

    return parser

def collect_data(num_data, env, agent, random_init_steps, epsilon, device, lz4_compress):
    data = []
    episode_rewards = []
    with tqdm.tqdm(total=num_data) as pbar:
        while pbar.n < num_data:
            state = env.reset()
            if lz4_compress:
                data.append((lz4.block.compress(state[-1]), None))
            else:
                data.append((state[-1], None))

            total_reward = 0

            # random initialization
            for _ in range(min(random_init_steps, num_data - pbar.n)):
                action = env.action_space.sample()
                state, reward, done, info = env.step(action)
                if lz4_compress:
                    data.append((lz4.block.compress(state[-1]), np.int8(action)))
                else:
                    data.append((state[-1], np.int8(action)))
                pbar.update()

                total_reward += reward

            done = False
            while not done and pbar.n < num_data:
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    state = torch.FloatTensor(np.asarray(state)).unsqueeze(0).to(device)
                    action = agent._act(state)

                state, reward, done, info = env.step(action)
                if lz4_compress:
                    data.append((lz4.block.compress(state[-1]), np.int8(action)))
                else:
                    data.append((state[-1], np.int8(action)))
                pbar.update()

                total_reward += reward

            if done:
                episode_rewards.append(total_reward)

    return data, len(episode_rewards), np.mean(episode_rewards)

def generate_data(num_data_rand, num_data_model, type, output_dir,
                  env, agent, random_init_steps, epsilon, device, lz4_compress):
    print(f"Start collecting {num_data_model} {type} data without random action selection")
    all_data, num_episodes, avg_reward = collect_data(
        num_data_model, env, agent, random_init_steps, 0, device, lz4_compress
    )
    print(f"Total full episodes: {num_episodes}, average reward per episode: {avg_reward}")

    print(f"Start collecting {num_data_rand} {type} data with random action selection")
    data, num_episodes, avg_reward = collect_data(
        num_data_rand, env, agent, random_init_steps, epsilon, device, lz4_compress
    )
    print(f"Total full episodes: {num_episodes}, average reward per episode: {avg_reward}")

    all_data.extend(data)
    del data

    data_file = output_dir / f"{type}_data.pkl"
    print(f"Saving {type} data to {data_file}")
    with open(data_file, "wb") as file:
        pickle.dump(all_data, file)
    print(f"Successfully saved {type} data to {data_file}")

    del all_data


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    if len(args.frame_size) == 1:  # type a single number in command line
        args.frame_size *= 2


    output_dir = pathlib.Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = preprocess_env(env,
        args.frame_size, args.skip_frames, args.stack_frames, args.lz4_compress
    )

    util.fix_random_seed(args.seed, env)

    agent_module = importlib.import_module("agents." + args.agent.lower())
    agent_class = getattr(agent_module, args.agent + "Test")
    agent = agent_class()

    agent.online_network.to(args.device)


    num_data_rand = int(args.num_train_data * args.random_act_ratio)
    num_data_model = args.num_train_data - num_data_rand
    generate_data(num_data_rand, num_data_model, "train", output_dir,
                  env, agent, args.random_init_steps, args.epsilon, args.device, args.lz4_compress)

    num_data_rand = int(args.num_eval_data * args.random_act_ratio)
    num_data_model = args.num_eval_data - num_data_rand
    generate_data(num_data_rand, num_data_model, "eval", output_dir,
                  env, agent, args.random_init_steps, args.epsilon, args.device, args.lz4_compress)


if __name__ == "__main__":
    main()
