import json
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
import logging
import sys
import argparse
import importlib

from env_preprocessor import preprocess_env
import util
from replay_buffer import NStepBuffer
from agents.agent_base import Agent

logger = logging.getLogger(__name__)  # set up a name so that matplotlib does not pollute the log
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%m-%d %H:%M:%S")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config")
    parser.add_argument("--agent", help="class name of the agent (case-sensitive)")
    parser.add_argument("--frame_size", type=int, nargs="*")
    parser.add_argument("--skip_frames", type=int)
    parser.add_argument("--stack_frames", type=int)
    parser.add_argument("--lz4_compress", action="store_true")
    parser.add_argument("--replay_buffer_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--adam_epsilon", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gamma", type=float, help="discount factor")
    parser.add_argument("--device")
    parser.add_argument("--steps_before_training", type=int)
    parser.add_argument("--steps_per_online_update", type=int)
    parser.add_argument("--steps_per_target_update", type=int)
    parser.add_argument("--total_episodes", type=int)
    parser.add_argument("--episodes_per_log", type=int)
    parser.add_argument("--episodes_per_save", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume_from_checkpoint", type=int, default=0)
    parser.add_argument("--n_step_td", type=int, default=1)
    parser.add_argument("--game_score_scalar", type=float, default=0)

    parser.add_argument("--epsilon_start", type=float, help="for epsilon-greedy")
    parser.add_argument("--epsilon_end", type=float, help="for epsilon-greedy")
    parser.add_argument("--epsilon_decay", type=float, help="for epsilon-greedy")

    parser.add_argument("--sigma0", type=float, help="for noisy network")
    parser.add_argument("--alpha", type=float, help="for prioritized replay buffer")
    parser.add_argument("--beta0", type=float, help="for prioritized replay buffer")

    parser.add_argument("--num_atoms", type=int, help="for distributional output")
    parser.add_argument("--v_min", type=float, help="for distributional output")
    parser.add_argument("--v_max", type=float, help="for distributional output")

    return parser


def main():
    parser = get_arg_parser()

    # precedence: command line argument > config file argument > default
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r") as file:
            vars(args).update(json.load(file))
    parser.parse_args(namespace=args)

    config = vars(args)

    frame_size = config["frame_size"]
    if isinstance(frame_size, int):  # type a single number in config file
        config["frame_size"] = (frame_size, frame_size)
    elif len(frame_size) == 1:  # type a single number in command line
        config["frame_size"] = frame_size * 2


    output_dir = pathlib.Path(config["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Training config: {config}")


    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = preprocess_env(env,
        config["frame_size"], config["skip_frames"], config["stack_frames"], config["lz4_compress"]
    )

    util.fix_random_seed(config["seed"], env)

    total_episodes = config["total_episodes"]
    steps_before_training = config["steps_before_training"]
    steps_per_online_update = config["steps_per_online_update"]
    steps_per_target_update = config["steps_per_target_update"]
    episodes_per_log = config["episodes_per_log"]
    episodes_per_save = config["episodes_per_save"]
    resume_from_checkpoint = config["resume_from_checkpoint"]
    game_score_scalar = config["game_score_scalar"]

    if resume_from_checkpoint:
        checkpoint_dir = output_dir / f"checkpoint-{resume_from_checkpoint}"
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
    else:
        checkpoint_dir = None

    agent_module = importlib.import_module("agents." + config["agent"].lower())
    agent_class = getattr(agent_module, config["agent"])
    agent: Agent = agent_class(
        (config["stack_frames"], *config["frame_size"]), env.action_space.n, config, checkpoint_dir
    )

    n_step_buffer = NStepBuffer(config["n_step_td"], config["gamma"], agent.replay_buffer)


    logger.info(f"Start collecting {steps_before_training} transitions")
    state = env.reset()
    game_score = 0
    for _ in range(steps_before_training):
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        additional_reward = (info["score"] - game_score) * game_score_scalar

        if len(agent.replay_buffer) == agent.replay_buffer.capacity // 2:
            logger.debug("replay buffer half full")
        elif len(agent.replay_buffer) == agent.replay_buffer.capacity - 1:
            logger.debug("replay buffer full")
        n_step_buffer.add(state, action, reward + additional_reward, next_state, done)

        if done:
            state = env.reset()
            game_score = 0
        else:
            state = next_state
            game_score = info["score"]


    reward_history = []
    game_score_history = []
    avg_reward_history = []
    avg_reward_history_file = output_dir / "avg_reward_history.txt"

    step_counter = 0

    lst_episode_steps = []

    logger.info("***** Start training *****")
    agent.start_training()
    for episode in range(resume_from_checkpoint + 1, resume_from_checkpoint + total_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        game_score = 0

        episode_steps = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            episode_steps += 1

            additional_reward = (info["score"] - game_score) * game_score_scalar

            if len(agent.replay_buffer) == agent.replay_buffer.capacity // 2:
                logger.debug("replay buffer half full")
            elif len(agent.replay_buffer) == agent.replay_buffer.capacity - 1:
                logger.debug("replay buffer full")
            n_step_buffer.add(state, action, reward + additional_reward, next_state, done)

            step_counter += 1
            if step_counter % steps_per_online_update == 0:
                agent.update_online()
            if step_counter % steps_per_target_update == 0:
                agent.update_target()

            state = next_state
            game_score = info["score"]
            total_reward += reward

        agent.on_episode_end(episode)

        reward_history.append(total_reward)
        game_score_history.append(info["score"])
        lst_episode_steps.append(episode_steps)

        if episode % episodes_per_log == 0:
            avg_reward = np.mean(reward_history)
            if game_score_scalar:
                logger.info(
                    f"Episode {episode}, Env Reward: {avg_reward}"
                    f", Game Score: {np.mean(game_score_history)}"
                )
            else:
                logger.info(f"Episode {episode}, Average Reward: {avg_reward}")
            logger.debug(f"avg step per episode: {np.mean(lst_episode_steps)}")
            lst_episode_steps = []

            avg_reward_history.append((episode, avg_reward))
            reward_history = []
            game_score_history = []

            agent.on_log(logger)

        if episode % episodes_per_save == 0:
            checkpoint_dir = output_dir / f"checkpoint-{episode}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            logger.info(f"Saving checkpoint to {checkpoint_dir}")
            agent.save(checkpoint_dir)

            with open(avg_reward_history_file, "a") as file:
                for epi, avg_reward in avg_reward_history:
                    file.write(f"{epi} {avg_reward}\n")
            avg_reward_history = []

            logger.info(f"Successfully saved checkpoint to {checkpoint_dir}")

    logger.info(f"Finish training after {step_counter} steps")


    with open(avg_reward_history_file, "r") as file:
        episodes = []
        rewards = []
        for line in file.readlines():
            line = line.split()
            episodes.append(int(line[0]))
            rewards.append(float(line[1]))

        plt.plot(episodes, rewards)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Training History")
        plt.savefig(output_dir / "training_history.png")


if __name__ == "__main__":
    main()
